"""Qwen-Video-Edit core: the video-token model_fn and the training module.

Architecture: source video -> (frozen) Wan 2.1 video VAE -> wan latents ->
trainable input projection -> Qwen-Image-Edit DiT (SFT or LoRA) with
grid/video positional encodings -> trainable output projection -> flow-match
loss in wan latent space. See README for the full writeup.
"""

import math

import torch

import rope_patch
rope_patch.apply()

from diffsynth.core import ModelConfig, gradient_checkpoint_forward
from diffsynth.diffusion import DiffusionTrainingModule
from diffsynth.pipelines.qwen_image import QwenImagePipeline, QwenImageUnit_PromptEmbedder

from projections import (
    FramewisePack4InProjection,
    FramewisePack4OutProjection,
    QwenToWanProjection,
    WanToQwenProjection,
)


def factorize_latent_grid(n: int) -> tuple[int, int]:
    for rows in range(int(math.isqrt(n)), 0, -1):
        if n % rows == 0:
            return rows, n // rows
    return 1, n


def num_token_groups(num_frames: int, latent_mode: str) -> int:
    if latent_mode == "qwen_framewise_pack4":
        assert num_frames % 4 == 0, f"pack4 needs num_frames % 4 == 0, got {num_frames}"
        return num_frames // 4
    assert (num_frames - 1) % 4 == 0, f"wan mode needs (num_frames-1) % 4 == 0, got {num_frames}"
    return (num_frames - 1) // 4 + 1


def model_fn_video_tokens(
    dit, in_proj, out_proj,
    latents, ref_latents, prompt_emb, prompt_emb_mask, timestep,
    latent_grid, zero_cond_t=False, pe_mode="grid",
    use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
):
    rows, cols = latent_grid
    B, C, T, h, w = latents.shape
    group = getattr(in_proj, "group", 1)
    n_tiles = T // group
    tok_h, tok_w = h // 2, w // 2

    if pe_mode == "video":
        # Video-DiT-style: each token group gets its own temporal index
        # (noise group i -> frame 2i, its reference -> 2i+1, preserving the
        # pretrained noise/edit "+1" relation per pair); spatial coords per
        # group like a standalone image.
        def pe_entries(base):
            return [{"frame": 2 * i + base, "height": tok_h, "width": tok_w,
                     "h_off": 0, "w_off": 0, "full_height": tok_h, "full_width": tok_w}
                    for i in range(n_tiles)]
    else:
        # Grid: all noise groups at frame 0 (refs at 1) with spatial offsets
        # in one big virtual image -- matches the image-grid pretraining.
        assert rows * cols == n_tiles, f"latent_grid {latent_grid} must tile {n_tiles} groups"

        def pe_entries(base):
            return [{"frame": base, "height": tok_h, "width": tok_w,
                     "h_off": (i // cols) * tok_h, "w_off": (i % cols) * tok_w,
                     "full_height": rows * tok_h, "full_width": cols * tok_w}
                    for i in range(n_tiles)]

    img_shapes = pe_entries(0) + pe_entries(1)
    txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
    timestep = timestep / 1000

    image = in_proj(latents)
    image_seq_len = image.shape[1]
    image = torch.cat([image, in_proj(ref_latents)], dim=1)

    if zero_cond_t:
        timestep = torch.cat([timestep, timestep * 0], dim=0)
        noise_len = sum(s["height"] * s["width"] for s in img_shapes[:n_tiles])
        cond_len = sum(s["height"] * s["width"] for s in img_shapes[n_tiles:])
        modulate_index = torch.tensor([[0] * noise_len + [1] * cond_len],
                                      device=timestep.device, dtype=torch.int)
    else:
        modulate_index = None

    conditioning = dit.time_text_embed(
        timestep, image.dtype,
        addition_t_cond=None if not dit.time_text_embed.use_additional_t_cond
        else torch.tensor([0]).to(device=image.device, dtype=torch.long))
    text = dit.txt_in(dit.txt_norm(prompt_emb))
    image_rotary_emb = dit.pos_embed(img_shapes, txt_seq_lens, device=latents.device)

    for block in dit.transformer_blocks:
        text, image = gradient_checkpoint_forward(
            block, use_gradient_checkpointing, use_gradient_checkpointing_offload,
            image=image, text=text, temb=conditioning,
            image_rotary_emb=image_rotary_emb, attention_mask=None,
            modulate_index=modulate_index)

    if zero_cond_t:
        conditioning = conditioning.chunk(2, dim=0)[0]
    image = dit.norm_out(image, conditioning)[:, :image_seq_len]
    return out_proj(image, num_frames=n_tiles, tokens_h=tok_h, tokens_w=tok_w)


class VideoTokenEditModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_id_with_origin_paths="Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors,"
                                   "Qwen/Qwen-Image:text_encoder/model*.safetensors",
        wan_vae_model_id_with_origin_path="Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth",
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        resume_from_checkpoint=None,
        num_frames=45, latent_mode="wan_compressed", pe_mode="grid",
        zero_cond_t=False, init_projections_from_dit=True, device="cpu",
    ):
        super().__init__()
        model_configs = self.parse_model_configs(None, model_id_with_origin_paths, device=device)
        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device=device, model_configs=model_configs,
            tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
            processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
        )

        assert latent_mode in ("wan_compressed", "qwen_framewise_pack4")
        self.latent_mode = latent_mode
        if latent_mode == "wan_compressed":
            wan_id, _, wan_pattern = wan_vae_model_id_with_origin_path.partition(":")
            pool = self.pipe.download_and_load_models(
                [ModelConfig(model_id=wan_id, origin_file_pattern=wan_pattern)], None)
            wan_vae = pool.fetch_model("wan_video_vae")
            wan_vae.requires_grad_(False)
            # Unregistered on purpose: keeps the frozen VAE out of any
            # DDP/DeepSpeed wrapper (its tiled forward has data-dependent
            # call counts, which must not emit collectives).
            object.__setattr__(self, "wan_vae", wan_vae)
        else:
            if self.pipe.vae is None:
                pool = self.pipe.download_and_load_models(
                    [ModelConfig(model_id="Qwen/Qwen-Image",
                                 origin_file_pattern="vae/diffusion_pytorch_model.safetensors")], None)
                self.pipe.vae = pool.fetch_model("qwen_image_vae")
            self.pipe.vae.requires_grad_(False)

        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            None, None, task="sft")

        inner = self.pipe.dit.img_in.out_features
        if latent_mode == "wan_compressed":
            self.in_proj = WanToQwenProjection(16, inner)
            self.out_proj = QwenToWanProjection(16, inner)
        else:
            self.in_proj = FramewisePack4InProjection(16, inner)
            self.out_proj = FramewisePack4OutProjection(16, inner)
        if init_projections_from_dit:
            self.in_proj.init_from_qwen_dit(self.pipe.dit)
            self.out_proj.init_from_qwen_dit(self.pipe.dit)
        self.in_proj.to(device=device, dtype=torch.bfloat16)
        self.out_proj.to(device=device, dtype=torch.bfloat16)

        self.resume_from_checkpoint(resume_from_checkpoint, None)

        self.prompt_unit = QwenImageUnit_PromptEmbedder()
        self.vae_batch_size = 24
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.zero_cond_t = zero_cond_t
        self.pe_mode = pe_mode
        self.latent_grid = factorize_latent_grid(num_token_groups(num_frames, latent_mode))
        print(f"[VideoTokenEditModule] latent_mode={latent_mode} pe_mode={pe_mode} "
              f"PE grid={self.latent_grid}")

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        device, dtype = self.pipe.device, self.pipe.torch_dtype
        if self.latent_mode == "wan_compressed":
            if next(self.wan_vae.parameters()).device != torch.device(device):
                self.wan_vae.to(device=device)
            tiled = video.shape[2] * video.shape[3] >= 700_000
            return self.wan_vae.encode([video.to(dtype=dtype)], device=device, tiled=tiled).to(device=device, dtype=dtype)
        frames = video.permute(1, 0, 2, 3).to(device=device, dtype=dtype)
        b = self.vae_batch_size
        latents = torch.cat([self.pipe.vae.encode(frames[i:i + b]) for i in range(0, frames.shape[0], b)], dim=0)
        return latents.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=dtype)

    @torch.no_grad()
    def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        device = self.pipe.device
        if self.latent_mode == "wan_compressed":
            if next(self.wan_vae.parameters()).device != torch.device(device):
                self.wan_vae.to(device=device)
            tiled = (latents.shape[3] * 8) * (latents.shape[4] * 8) >= 700_000
            return self.wan_vae.decode(latents, device=device, tiled=tiled)[0].cpu()
        frames_lat = latents[0].permute(1, 0, 2, 3)
        b = self.vae_batch_size
        frames = torch.cat([self.pipe.vae.decode(frames_lat[i:i + b]) for i in range(0, frames_lat.shape[0], b)], dim=0)
        return frames.permute(1, 0, 2, 3).cpu()

    def forward(self, data, inputs=None):
        pipe = self.pipe
        device, dtype = pipe.device, pipe.torch_dtype

        with torch.no_grad():
            ref_latents = self.encode_video(data["source_video"])
            target_latents = self.encode_video(data["target_video"])
            emb = self.prompt_unit.process(pipe, prompt=data["prompt"], edit_image=data["preview_image"])

        timestep_id = torch.randint(0, len(pipe.scheduler.timesteps), (1,))
        timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=dtype, device=device)
        noise = torch.randn_like(target_latents)
        noisy_latents = pipe.scheduler.add_noise(target_latents, noise, timestep)
        training_target = pipe.scheduler.training_target(target_latents, noise, timestep)

        noise_pred = model_fn_video_tokens(
            pipe.dit, self.in_proj, self.out_proj,
            latents=noisy_latents, ref_latents=ref_latents,
            prompt_emb=emb["prompt_emb"], prompt_emb_mask=emb["prompt_emb_mask"],
            timestep=timestep, latent_grid=self.latent_grid,
            zero_cond_t=self.zero_cond_t, pe_mode=self.pe_mode,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload)

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        return loss * pipe.scheduler.training_weight(timestep)
