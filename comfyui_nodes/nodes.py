"""ComfyUI custom nodes for Qwen-Video-Edit.

Single-chunk semantics: the sampler edits ONE window of `num_frames` frames
(pad-with-last if fewer are supplied, truncate if more). For long videos,
feed chunks one at a time (e.g. VHS LoadVideo with frame_load_cap /
skip_first_frames) and concatenate the outputs.

All heavy lifting reuses the repo's own modules (model.py / projections.py /
dataset.py and the vendored diffsynth/ and wan/ packages); models are loaded
lazily on first execution, never at ComfyUI startup.
"""

import os
import sys

# The repo root doubles as the custom-node package root: make its top-level
# modules (model, projections, dataset, rope_patch, diffsynth, wan)
# importable, shadowing any pip-installed diffsynth.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_LOADER_CACHE = {}   # params-key -> editing stack dict
_WAN22_CACHE = {}    # params-key -> WanT2V
_LOW_RAM = False     # set by the enhance node; True = unload instead of parking in RAM


def _editing_stack_to(stack, device):
    """Move the whole Qwen editing stack between GPU and CPU RAM. The editing
    stack (~55GB) and the Wan2.2 low-noise expert (~28GB) cannot coexist in
    80GB of VRAM, so the sampler and the enhance node swap them."""
    import torch
    pipe = stack["pipe"]
    for m in (getattr(pipe, "dit", None), getattr(pipe, "text_encoder", None),
              getattr(pipe, "vae", None), stack.get("wan_vae"),
              stack.get("in_proj"), stack.get("out_proj")):
        if m is not None and hasattr(m, "to"):
            try:
                m.to(device)
            except Exception:  # noqa: BLE001 -- non-module wrappers
                pass
    if device == "cpu":
        torch.cuda.empty_cache()


def _wan22_to_cpu():
    import torch
    for wan_t2v in _WAN22_CACHE.values():
        for name in ("low_noise_model", "high_noise_model"):
            m = getattr(wan_t2v, name, None)
            if m is not None and hasattr(m, "to"):
                m.to("cpu")
    torch.cuda.empty_cache()


def _load_models_into(stack):
    """(Re)load the editing stack's models from disk into stack (on GPU).
    stack["params"] = (checkpoint, latent_mode, pe_mode, zero_cond_t,
    model_id_with_origin_paths)."""
    import torch
    import rope_patch
    rope_patch.apply()
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.qwen_image import QwenImagePipeline
    from projections import (FramewisePack4InProjection, FramewisePack4OutProjection,
                             QwenToWanProjection, WanToQwenProjection)
    from infer import load_checkpoint_into

    checkpoint, latent_mode, _, _, model_id_with_origin_paths = stack["params"]
    device, dtype = "cuda", torch.bfloat16
    model_configs = [ModelConfig(model_id=e.partition(":")[0], origin_file_pattern=e.partition(":")[2])
                     for e in model_id_with_origin_paths.split(",")]
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=dtype, device=device, model_configs=model_configs,
        tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"))
    wan_vae = None
    if latent_mode == "wan_compressed":
        pool = pipe.download_and_load_models(
            [ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth")], None)
        wan_vae = pool.fetch_model("wan_video_vae")
        in_proj = WanToQwenProjection(16, pipe.dit.img_in.out_features)
        out_proj = QwenToWanProjection(16, pipe.dit.img_in.out_features)
    else:
        if pipe.vae is None:
            pool = pipe.download_and_load_models(
                [ModelConfig(model_id="Qwen/Qwen-Image",
                             origin_file_pattern="vae/diffusion_pytorch_model.safetensors")], None)
            pipe.vae = pool.fetch_model("qwen_image_vae")
        in_proj = FramewisePack4InProjection(16, pipe.dit.img_in.out_features)
        out_proj = FramewisePack4OutProjection(16, pipe.dit.img_in.out_features)
    load_checkpoint_into(pipe, in_proj, out_proj, checkpoint)
    in_proj.to(device=device, dtype=dtype)
    out_proj.to(device=device, dtype=dtype)
    stack.update({"pipe": pipe, "wan_vae": wan_vae, "in_proj": in_proj, "out_proj": out_proj})


def _unload_editing_stack(stack):
    """Truly free the editing stack (VRAM *and* host RAM); the sampler
    reloads it from disk on its next run via stack["params"]."""
    import gc
    import torch
    for k in ("pipe", "wan_vae", "in_proj", "out_proj"):
        stack.pop(k, None)
    gc.collect()
    torch.cuda.empty_cache()


def _comfy_images_to_frames(images):
    """ComfyUI IMAGE (B,H,W,C float 0..1) -> list of HWC uint8 numpy frames."""
    import torch
    return [(img.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy() for img in images]


def _video_tensor_to_comfy_images(video):
    """(C,T,H,W) in [-1,1] -> ComfyUI IMAGE (T,H,W,C float 0..1, cpu)."""
    return ((video.float().clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).contiguous().cpu()


class QwenVideoEditLoader:
    """Loads the editing stack: Qwen-Image-Edit DiT + text encoder, the
    Wan2.1 VAE (or Qwen image VAE for pack4), the two bridge projections,
    and your fine-tuned checkpoint. latent_mode / pe_mode / zero_cond_t MUST
    match how the checkpoint was trained."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "checkpoint": ("STRING", {"default": "checkpoints/360P/step-30000.safetensors"}),
            "latent_mode": (["wan_compressed", "qwen_framewise_pack4"],),
            "pe_mode": (["grid", "video"],),
            "zero_cond_t": ("BOOLEAN", {"default": False}),
            "model_id_with_origin_paths": ("STRING", {
                "default": "Qwen/Qwen-Image-Edit:transformer/diffusion_pytorch_model*.safetensors,"
                           "Qwen/Qwen-Image:text_encoder/model*.safetensors"}),
        }}

    RETURN_TYPES = ("QVE_MODEL",)
    RETURN_NAMES = ("qve_model",)
    FUNCTION = "load"
    CATEGORY = "QwenVideoEdit"

    def load(self, checkpoint, latent_mode, pe_mode, zero_cond_t, model_id_with_origin_paths):
        key = (checkpoint, latent_mode, pe_mode, zero_cond_t, model_id_with_origin_paths)
        if key in _LOADER_CACHE:
            return (_LOADER_CACHE[key],)
        stack = {"params": key, "latent_mode": latent_mode, "pe_mode": pe_mode,
                 "zero_cond_t": zero_cond_t}
        _load_models_into(stack)
        _LOADER_CACHE.clear()   # keep at most one editing stack around
        _LOADER_CACHE[key] = stack
        return (stack,)


class QwenVideoEditSampler:
    """Edits ONE chunk of frames with one instruction. Frames beyond
    num_frames are ignored; fewer frames are padded with the last one.
    max_pixels > 0 resizes adaptively (aspect-preserving, /16); set it to
    match the resolution the checkpoint was trained for."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "qve_model": ("QVE_MODEL",),
            "images": ("IMAGE",),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "negative_prompt": ("STRING", {"default": " "}),
            "num_frames": ("INT", {"default": 45, "min": 4, "max": 241}),
            "max_pixels": ("INT", {"default": 245760, "min": 0, "max": 2 ** 24,
                                   "tooltip": "0 = keep input resolution (rounded to /16)"}),
            "steps": ("INT", {"default": 40, "min": 1, "max": 200}),
            "cfg_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 15.0, "step": 0.5}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2 ** 32 - 1}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "sample"
    CATEGORY = "QwenVideoEdit"

    def sample(self, qve_model, images, prompt, negative_prompt, num_frames,
               max_pixels, steps, cfg_scale, seed):
        import torch
        from dataset import adaptive_dims, build_preview_grid, first_n_frames_padded, frames_to_tensor
        from model import factorize_latent_grid, model_fn_video_tokens, num_token_groups
        from diffsynth.pipelines.qwen_image import QwenImageUnit_PromptEmbedder

        # Swap: get the Wan2.2 experts off the GPU (dropped entirely in
        # low-RAM mode), bring the editing stack (back) onto the GPU --
        # reloading it from disk if the enhance node unloaded it.
        if _LOW_RAM:
            _WAN22_CACHE.clear()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        else:
            _wan22_to_cpu()
        if "pipe" not in qve_model:
            print("[qwen-video-edit] Reloading the editing stack from disk (low_ram mode) ...")
            _load_models_into(qve_model)
        else:
            _editing_stack_to(qve_model, "cuda")

        pipe, wan_vae = qve_model["pipe"], qve_model["wan_vae"]
        in_proj, out_proj = qve_model["in_proj"], qve_model["out_proj"]
        latent_mode, pe_mode = qve_model["latent_mode"], qve_model["pe_mode"]
        device, dtype = "cuda", torch.bfloat16

        frames = _comfy_images_to_frames(images)
        chunk = first_n_frames_padded(frames[:num_frames], num_frames)
        h0, w0 = chunk[0].shape[:2]
        if max_pixels > 0:
            width, height = adaptive_dims(w0, h0, max_pixels)
        else:
            width, height = w0 // 16 * 16, h0 // 16 * 16
        latent_grid = factorize_latent_grid(num_token_groups(num_frames, latent_mode))

        with torch.no_grad():
            source = frames_to_tensor(chunk, width, height)
            preview = build_preview_grid(chunk)
            if latent_mode == "wan_compressed":
                tiled = height * width >= 700_000
                ref_latents = wan_vae.encode([source.to(dtype=dtype)], device=device,
                                             tiled=tiled).to(device=device, dtype=dtype)
            else:
                frames_dev = source.permute(1, 0, 2, 3).to(device=device, dtype=dtype)
                lat = torch.cat([pipe.vae.encode(frames_dev[i:i + 24])
                                 for i in range(0, frames_dev.shape[0], 24)], dim=0)
                ref_latents = lat.permute(1, 0, 2, 3).unsqueeze(0)

            emb = QwenImageUnit_PromptEmbedder().process(pipe, prompt=prompt, edit_image=preview)
            use_cfg = cfg_scale > 1.0
            if use_cfg:
                neg_emb = QwenImageUnit_PromptEmbedder().process(
                    pipe, prompt=negative_prompt, edit_image=preview)

            group = getattr(in_proj, "group", 1)
            noise_seq_len = (ref_latents.shape[2] // group) * (ref_latents.shape[3] // 2) * (ref_latents.shape[4] // 2)
            pipe.scheduler.set_timesteps(steps, dynamic_shift_len=noise_seq_len)
            gen = torch.Generator(device="cpu").manual_seed(seed)
            latents = torch.randn(ref_latents.shape, generator=gen).to(device=device, dtype=dtype)

            def forward(e, timestep):
                return model_fn_video_tokens(
                    pipe.dit, in_proj, out_proj, latents=latents, ref_latents=ref_latents,
                    prompt_emb=e["prompt_emb"], prompt_emb_mask=e["prompt_emb_mask"],
                    timestep=timestep, latent_grid=latent_grid,
                    zero_cond_t=qve_model["zero_cond_t"], pe_mode=pe_mode)

            try:
                import comfy.utils
                pbar = comfy.utils.ProgressBar(steps)
            except Exception:
                pbar = None
            for pid, timestep in enumerate(pipe.scheduler.timesteps):
                timestep = timestep.unsqueeze(0).to(dtype=dtype, device=device)
                noise_pred = forward(emb, timestep)
                if use_cfg:
                    neg_pred = forward(neg_emb, timestep)
                    comb = neg_pred + cfg_scale * (noise_pred - neg_pred)
                    comb = comb * (torch.norm(noise_pred, dim=1, keepdim=True)
                                   / torch.norm(comb, dim=1, keepdim=True))
                    noise_pred = comb
                latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[pid], latents)
                if pbar:
                    pbar.update(1)

            if latent_mode == "wan_compressed":
                tiled = (latents.shape[3] * 8) * (latents.shape[4] * 8) >= 700_000
                video = wan_vae.decode(latents, device=device, tiled=tiled)[0].cpu()
            else:
                fl = latents[0].permute(1, 0, 2, 3)
                video = torch.cat([pipe.vae.decode(fl[i:i + 24])
                                   for i in range(0, fl.shape[0], 24)], dim=0).permute(1, 0, 2, 3).cpu()

        return (_video_tensor_to_comfy_images(video),)


class Wan22Enhance:
    """Ditto-style denoising enhancement: re-noise the edited chunk a few
    steps and denoise with Wan2.2-T2V-A14B. Uses the same prompt as the
    edit. offload=True keeps the 14B experts in CPU RAM and swaps them to
    GPU per forward -- recommended inside ComfyUI, where the Qwen editing
    stack is still resident in VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "wan22_ckpt_dir": ("STRING", {"default": "/models/Wan-AI/Wan2.2-T2V-A14B"}),
            "enhance_steps": ("INT", {"default": 4, "min": 1, "max": 20}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2 ** 32 - 1}),
            "offload": ("BOOLEAN", {"default": True}),
            "low_ram": ("BOOLEAN", {"default": False, "tooltip":
                "True: FREE the Qwen editing stack before enhancing instead of "
                "parking it in CPU RAM (for hosts with < ~100GB RAM). The next "
                "sampler run reloads ~55GB of weights from disk."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "enhance"
    CATEGORY = "QwenVideoEdit"

    def enhance(self, images, prompt, wan22_ckpt_dir, enhance_steps, seed, offload, low_ram):
        import torch
        import wan as wan_pkg
        from wan.configs import WAN_CONFIGS

        # Swap: the Qwen editing stack cannot share 80GB with the Wan2.2
        # expert. Default: park it in CPU RAM (fast to bring back).
        # low_ram: free it entirely -- the sampler reloads it from disk.
        global _LOW_RAM
        _LOW_RAM = low_ram
        for _stack in _LOADER_CACHE.values():
            if low_ram:
                _unload_editing_stack(_stack)
            else:
                _editing_stack_to(_stack, "cpu")

        cfg = WAN_CONFIGS["t2v-A14B"]
        skip_high = enhance_steps < 12
        key = (wan22_ckpt_dir, skip_high)
        if key not in _WAN22_CACHE:
            _WAN22_CACHE.clear()
            _WAN22_CACHE[key] = wan_pkg.WanT2V(
                config=cfg, checkpoint_dir=wan22_ckpt_dir, device_id=0, rank=0,
                t5_cpu=True, convert_model_dtype=True, skip_high_noise_model=skip_high)
        wan_t2v = _WAN22_CACHE[key]

        # IMAGE (T,H,W,C 0..1) -> (C,T,H,W) [-1,1]
        video = images.permute(3, 0, 1, 2).float() * 2 - 1
        with torch.no_grad():
            enhanced = wan_t2v.generate(
                prompt, size=(video.shape[2], video.shape[3]),
                shift=cfg.sample_shift, sample_solver="unipc",
                sampling_steps=cfg.sample_steps, guide_scale=cfg.sample_guide_scale,
                seed=seed, offload_model=offload,
                input_video=video.to("cuda"),
                forward_step=enhance_steps, skip_backward_step=enhance_steps)
        return (_video_tensor_to_comfy_images(enhanced.cpu()),)


NODE_CLASS_MAPPINGS = {
    "QwenVideoEditLoader": QwenVideoEditLoader,
    "QwenVideoEditSampler": QwenVideoEditSampler,
    "QVEWan22Enhance": Wan22Enhance,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVideoEditLoader": "Qwen-Video-Edit Loader",
    "QwenVideoEditSampler": "Qwen-Video-Edit Sampler (one chunk)",
    "QVEWan22Enhance": "Wan2.2 Enhance (Ditto)",
}
