"""Long-video editing pipeline: one prompt per line edits one --num_frames
window of the source video, in order; stops when either prompts or frames run
out. Every chunk is then refined by Wan2.2 denoising-enhancement (Ditto) --
chained IN MEMORY, so no intermediate un-enhanced files are written and only
the final results land on disk.

    python infer.py \
        --source_video input.mp4 --prompts_file prompts.txt \
        --checkpoint checkpoints/step-20000.safetensors \
        --wan22_ckpt_dir /models/Wan-AI/Wan2.2-T2V-A14B \
        --num_frames 45 --video_max_pixels 245760 --zero_cond_t \
        --output_dir ./out

Outputs {output_dir}/{name}_chunk000.mp4/.txt, chunk001... Concat with the
ffmpeg command printed at the end.
"""

import argparse
import os

import functools
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Make progress visible even when stdout is piped/not a tty (kubectl logs,
# nohup, tee): line-buffered prints everywhere.
print = functools.partial(print, flush=True)

import rope_patch
rope_patch.apply()

from diffsynth.core import ModelConfig
from diffsynth.pipelines.qwen_image import QwenImagePipeline, QwenImageUnit_PromptEmbedder
from safetensors.torch import load_file

from dataset import (
    adaptive_dims, build_preview_grid, decode_video_frames,
    first_n_frames_padded, frames_to_tensor, open_maybe_remote,
)
from model import factorize_latent_grid, model_fn_video_tokens, num_token_groups
from projections import (
    FramewisePack4InProjection, FramewisePack4OutProjection,
    QwenToWanProjection, WanToQwenProjection,
)


def save_video(frames_uint8, path_base, fps=16):
    import imageio.v3 as iio
    path = path_base + ".mp4"
    iio.imwrite(path, frames_uint8, fps=fps, codec="libx264")
    return path


def tensor_to_uint8_frames(video):  # (C,T,H,W) [-1,1] -> list of HWC uint8
    v = ((video.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    return [v[:, t].permute(1, 2, 0).cpu().numpy() for t in range(v.shape[1])]


def load_checkpoint_into(pipe, in_proj, out_proj, checkpoint_path):
    state_dict = load_file(checkpoint_path)
    dit_sd = {k[len("pipe.dit."):]: v for k, v in state_dict.items() if k.startswith("pipe.dit.")}
    in_proj.load_state_dict({k[len("in_proj."):]: v for k, v in state_dict.items() if k.startswith("in_proj.")})
    out_proj.load_state_dict({k[len("out_proj."):]: v for k, v in state_dict.items() if k.startswith("out_proj.")})
    if any("lora" in k for k in dit_sd):
        pipe.load_lora(pipe.dit, state_dict=dit_sd, hotload=True)
        print(f"[infer] Loaded LoRA DiT weights ({len(dit_sd)} tensors) + projections.")
    else:
        pipe.dit.load_state_dict(dit_sd, strict=False)
        print(f"[infer] Loaded full DiT weights ({len(dit_sd)} tensors) + projections.")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source_video", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, required=True, help="One edit prompt per line.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./out")
    parser.add_argument("--model_id_with_origin_paths", type=str,
                        default="Qwen/Qwen-Image-Edit:transformer/diffusion_pytorch_model*.safetensors,"
                                "Qwen/Qwen-Image:text_encoder/model*.safetensors")
    parser.add_argument("--wan_vae_model_id_with_origin_path", type=str,
                        default="Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth")
    parser.add_argument("--num_frames", type=int, default=45)
    parser.add_argument("--video_height", type=int, default=384)
    parser.add_argument("--video_width", type=int, default=640)
    parser.add_argument("--video_max_pixels", type=int, default=0)
    parser.add_argument("--latent_mode", type=str, default="wan_compressed",
                        choices=["wan_compressed", "qwen_framewise_pack4"], help="MUST match the checkpoint.")
    parser.add_argument("--pe_mode", type=str, default="grid", choices=["grid", "video"], help="MUST match the checkpoint.")
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--cfg_scale", type=float, default=4.0,
                        help="True-CFG scale for the editing stage. 1 (default) = single "
                             "conditional forward, matching how the model was evaluated "
                             "during training; >1 doubles compute per step.")
    parser.add_argument("--negative_prompt", type=str, default=" ")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--zero_cond_t", default=False, action="store_true", help="MUST match the checkpoint.")
    # Enhancement (Wan2.2 denoising-enhancement, Ditto) -- an integral stage
    # of the pipeline. The vendored ./wan package is used; --skip_enhance
    # exists only for debugging the editing stage in isolation.
    parser.add_argument("--wan22_ckpt_dir", type=str, default=None,
                        help="Path to Wan-AI/Wan2.2-T2V-A14B (required unless --skip_enhance).")
    parser.add_argument("--enhance_steps", type=int, default=4, help="Ditto forward/backward steps.")
    parser.add_argument("--skip_enhance", default=False, action="store_true",
                        help="DEBUG ONLY: write the raw edited chunks without Wan2.2 enhancement.")
    parser.add_argument("--wan22_offload", default=False, action="store_true",
                        help="Keep Wan2.2 experts in CPU RAM and swap to GPU per forward "
                             "(needs ~70GB+ host RAM but less VRAM headroom). Default OFF: "
                             "experts stay on the GPU -- at <=480p/45f everything fits in "
                             "80GB and host RAM stays small.")
    args = parser.parse_args()
    args.enhance = not args.skip_enhance
    if args.enhance:
        assert args.wan22_ckpt_dir, "--wan22_ckpt_dir is required (or pass --skip_enhance for debugging)"

    device, dtype = "cuda", torch.bfloat16
    os.makedirs(args.output_dir, exist_ok=True)
    prompts = [l.strip() for l in open(args.prompts_file) if l.strip()]
    with open_maybe_remote(args.source_video) as f:
        all_frames = decode_video_frames(f.read())
    n_chunks = min(len(prompts), max(1, (len(all_frames) + args.num_frames - 1) // args.num_frames))
    n_chunks = min(n_chunks, len([s for s in range(0, len(all_frames), args.num_frames)]))
    print(f"[infer] {len(all_frames)} frames, {len(prompts)} prompts -> {n_chunks} chunks")

    latent_grid = factorize_latent_grid(num_token_groups(args.num_frames, args.latent_mode))
    base_name = os.path.splitext(os.path.basename(args.source_video))[0]

    # ---- Stage 1: Qwen editing (results kept in CPU memory) ----------------
    print("[infer] Stage 1/2: loading the Qwen editing stack (DiT ~40GB + text "
          "encoder ~15GB; FIRST run also downloads them into "
          f"{os.environ.get('DIFFSYNTH_MODEL_BASE_PATH', './models')} -- can take a while) ...")
    t0 = time.time()
    model_configs = [ModelConfig(model_id=e.partition(":")[0], origin_file_pattern=e.partition(":")[2])
                     for e in args.model_id_with_origin_paths.split(",")]
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=dtype, device=device, model_configs=model_configs,
        tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"))
    wan_vae = None
    if args.latent_mode == "wan_compressed":
        wan_id, _, wan_pattern = args.wan_vae_model_id_with_origin_path.partition(":")
        pool = pipe.download_and_load_models([ModelConfig(model_id=wan_id, origin_file_pattern=wan_pattern)], None)
        wan_vae = pool.fetch_model("wan_video_vae")
        in_proj, out_proj = WanToQwenProjection(16, pipe.dit.img_in.out_features), QwenToWanProjection(16, pipe.dit.img_in.out_features)
    else:
        if pipe.vae is None:
            pool = pipe.download_and_load_models(
                [ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors")], None)
            pipe.vae = pool.fetch_model("qwen_image_vae")
        in_proj, out_proj = FramewisePack4InProjection(16, pipe.dit.img_in.out_features), FramewisePack4OutProjection(16, pipe.dit.img_in.out_features)
    load_checkpoint_into(pipe, in_proj, out_proj, args.checkpoint)
    in_proj.to(device=device, dtype=dtype)
    out_proj.to(device=device, dtype=dtype)
    print(f"[infer] Editing stack ready in {time.time() - t0:.0f}s.")

    # Edited chunks are SPILLED TO DISK (one small .pt per chunk, deleted
    # after enhancement) instead of accumulating in RAM -- keeps host memory
    # O(1) regardless of video length (a decoded 45f/0.25MP chunk is ~70MB;
    # an hour-long video would otherwise eat >100GB of RAM).
    tmp_dir = os.path.join(args.output_dir, ".tmp_edited")
    if args.enhance:
        os.makedirs(tmp_dir, exist_ok=True)
    edited_ids = []
    for cid in range(n_chunks):
        chunk = first_n_frames_padded(all_frames[cid * args.num_frames:(cid + 1) * args.num_frames], args.num_frames)
        prompt = prompts[cid]
        print(f"[infer] === chunk {cid}: {prompt[:80]}")
        if args.video_max_pixels:
            h0, w0 = chunk[0].shape[:2]
            width, height = adaptive_dims(w0, h0, args.video_max_pixels)
        else:
            width, height = args.video_width, args.video_height
        source = frames_to_tensor(chunk, width, height)
        preview = build_preview_grid(chunk)

        if args.latent_mode == "wan_compressed":
            tiled = height * width >= 700_000
            ref_latents = wan_vae.encode([source.to(dtype=dtype)], device=device, tiled=tiled).to(device=device, dtype=dtype)
        else:
            frames_dev = source.permute(1, 0, 2, 3).to(device=device, dtype=dtype)
            lat = torch.cat([pipe.vae.encode(frames_dev[i:i + 24]) for i in range(0, frames_dev.shape[0], 24)], dim=0)
            ref_latents = lat.permute(1, 0, 2, 3).unsqueeze(0)
        emb = QwenImageUnit_PromptEmbedder().process(pipe, prompt=prompt, edit_image=preview)
        use_cfg = args.cfg_scale > 1.0
        if use_cfg:
            neg_emb = QwenImageUnit_PromptEmbedder().process(
                pipe, prompt=args.negative_prompt, edit_image=preview)

        group = getattr(in_proj, "group", 1)
        noise_seq_len = (ref_latents.shape[2] // group) * (ref_latents.shape[3] // 2) * (ref_latents.shape[4] // 2)
        pipe.scheduler.set_timesteps(args.num_inference_steps, dynamic_shift_len=noise_seq_len)
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        latents = torch.randn(ref_latents.shape, generator=gen).to(device=device, dtype=dtype)

        def forward(e, timestep):
            return model_fn_video_tokens(
                pipe.dit, in_proj, out_proj, latents=latents, ref_latents=ref_latents,
                prompt_emb=e["prompt_emb"], prompt_emb_mask=e["prompt_emb_mask"],
                timestep=timestep, latent_grid=latent_grid,
                zero_cond_t=args.zero_cond_t, pe_mode=args.pe_mode)

        for pid, timestep in enumerate(tqdm(pipe.scheduler.timesteps, desc=f"chunk {cid} denoise")):
            timestep = timestep.unsqueeze(0).to(dtype=dtype, device=device)
            noise_pred = forward(emb, timestep)
            if use_cfg:
                # True CFG (negative prompt, same ref latents) with
                # norm-preserving rescale over the latent channel dim.
                neg_pred = forward(neg_emb, timestep)
                comb = neg_pred + args.cfg_scale * (noise_pred - neg_pred)
                comb = comb * (torch.norm(noise_pred, dim=1, keepdim=True)
                               / torch.norm(comb, dim=1, keepdim=True))
                noise_pred = comb
            latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[pid], latents)

        if args.latent_mode == "wan_compressed":
            tiled = (latents.shape[3] * 8) * (latents.shape[4] * 8) >= 700_000
            video = wan_vae.decode(latents, device=device, tiled=tiled)[0].cpu()
        else:
            fl = latents[0].permute(1, 0, 2, 3)
            video = torch.cat([pipe.vae.decode(fl[i:i + 24]) for i in range(0, fl.shape[0], 24)], dim=0).permute(1, 0, 2, 3).cpu()

        if args.enhance:
            torch.save({"prompt": prompt, "video": video},
                       os.path.join(tmp_dir, f"chunk{cid:03d}.pt"))
            edited_ids.append(cid)
        else:
            out_base = os.path.join(args.output_dir, f"{base_name}_chunk{cid:03d}")
            save_video(tensor_to_uint8_frames(video), out_base, fps=args.fps)
            open(out_base + ".txt", "w").write(prompt)
            print(f"[infer] Wrote {out_base}.mp4")
        del video, latents, ref_latents

    if args.enhance:
        # ---- Stage 2: free the Qwen stack, load Wan2.2, enhance in memory --
        del pipe, in_proj, out_proj, wan_vae
        torch.cuda.empty_cache()
        print("[infer] Stage 2/2: Qwen stack freed; loading Wan2.2 for enhancement "
              "(two 14B experts, several minutes to load) ...")
        t1 = time.time()
        import wan  # Ditto's package (see README)
        from wan.configs import WAN_CONFIGS
        cfg = WAN_CONFIGS["t2v-A14B"]
        # Expert selection is t >= boundary (875): with the ~4-step
        # enhancement all timesteps are far below it, so the high-noise
        # expert would never run -- skip loading it (saves ~28GB + minutes).
        # It only becomes relevant for enhance_steps >= ~15 (t crosses 875).
        skip_high = args.enhance_steps < 12
        wan_t2v = wan.WanT2V(config=cfg, checkpoint_dir=args.wan22_ckpt_dir, device_id=0, rank=0,
                             t5_cpu=True, convert_model_dtype=True,
                             skip_high_noise_model=skip_high)
        print(f"[infer] Wan2.2 ready in {time.time() - t1:.0f}s; enhancing {len(edited_ids)} chunks ...")
        for cid in tqdm(edited_ids, desc="enhance"):
            pt_path = os.path.join(tmp_dir, f"chunk{cid:03d}.pt")
            blob = torch.load(pt_path)
            prompt, video = blob["prompt"], blob["video"]
            enhanced = wan_t2v.generate(
                prompt, size=(video.shape[2], video.shape[3]),
                shift=cfg.sample_shift, sample_solver="unipc",
                sampling_steps=cfg.sample_steps, guide_scale=cfg.sample_guide_scale,
                seed=args.seed, offload_model=args.wan22_offload,
                input_video=video.to("cuda"),
                forward_step=args.enhance_steps, skip_backward_step=args.enhance_steps)
            out_base = os.path.join(args.output_dir, f"{base_name}_chunk{cid:03d}")
            save_video(tensor_to_uint8_frames(enhanced.cpu()), out_base, fps=args.fps)
            open(out_base + ".txt", "w").write(prompt)
            os.remove(pt_path)  # temp spill no longer needed
            print(f"[infer] Wrote (enhanced) {out_base}.mp4")
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    print(f"[infer] Concat: cd {args.output_dir} && printf 'file %s\\n' {base_name}_chunk*.mp4 > list.txt && "
          f"ffmpeg -f concat -safe 0 -i list.txt -c copy {base_name}_edited_full.mp4")


if __name__ == "__main__":
    main()
