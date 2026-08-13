"""Empirical step-by-step verification with the STOCK Qwen-Image-Edit
checkpoint (no fine-tuning) -- the observations that motivated this project.

    python empirical.py --mode image_grid  --frames_dir ./frames --prompt "..."
    python empirical.py --mode latent_grid --frames_dir ./frames --prompt "..."

image_grid : tile N frames into ONE image and edit it with the unmodified
             Qwen-Image-Edit pipeline. Demonstrates that an image-edit model
             can already edit "video as a contact sheet".
latent_grid: encode each frame SEPARATELY with Qwen's image VAE, feed the
             DiT the concatenated tokens with grid positional encodings
             (each frame keeps its spatial offset inside the virtual grid;
             requires this repo's RoPE patch), then decode per frame.
             Demonstrates that the PE treatment transfers zero-shot -- the
             bridge between image grids and per-frame video latents.
"""

import argparse
import math
import os

import numpy as np
import torch
from PIL import Image

import rope_patch
rope_patch.apply()

from diffsynth.core import ModelConfig
from diffsynth.pipelines.qwen_image import QwenImagePipeline, QwenImageUnit_PromptEmbedder


def load_frames(frames_dir, n, uniform_sample=False):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(exts))
    assert len(files) >= n, f"need >= {n} frames in {frames_dir}, found {len(files)}"
    if uniform_sample:
        # Uniformly spread keyframes over the whole directory (first and last
        # included) instead of just taking the first n frames.
        idx = np.linspace(0, len(files) - 1, n).round().astype(int)
        files = [files[i] for i in idx]
        print(f"[empirical] uniform-sampled frames: {files}")
    else:
        files = files[:n]
    return [Image.open(os.path.join(frames_dir, f)).convert("RGB") for f in files]


def make_grid(frames, rows, cols, tile_w, tile_h):
    grid = Image.new("RGB", (tile_w * cols, tile_h * rows))
    for i, fr in enumerate(frames):
        grid.paste(fr.resize((tile_w, tile_h)), ((i % cols) * tile_w, (i // cols) * tile_h))
    return grid


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", type=str, required=True, choices=["image_grid", "latent_grid"])
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory of frames (sorted by name).")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--tile_size", type=str, default="256x448",
                        help="HxW per tile, multiples of 32 (even token dims). Keep the full "
                             "canvas rows*H x cols*W near 1024x1024 total -- Qwen-Image-Edit "
                             "degrades sharply beyond that.")
    parser.add_argument("--uniform_sample", action="store_true",
                        help="Uniformly sample the rows*cols keyframes across ALL frames in "
                             "--frames_dir (default: take the first rows*cols frames).")
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--negative_prompt", type=str, default=" ")
    parser.add_argument("--cfg_scale", type=float, default=4.0,
                        help="Classifier-free guidance scale (1 disables the negative pass).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero_cond_t", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--output_dir", type=str, default="./empirical_out")
    args = parser.parse_args()

    device, dtype = "cuda", torch.bfloat16
    os.makedirs(args.output_dir, exist_ok=True)
    tile_h, tile_w = (int(x) for x in args.tile_size.lower().split("x"))
    assert tile_h % 32 == 0 and tile_w % 32 == 0, \
        "--tile_size must be multiples of 32 (token dims must be even for centered RoPE)"
    canvas_area = tile_h * args.rows * tile_w * args.cols
    if canvas_area > 1.3 * 1024 * 1024:
        print(f"[empirical] WARNING: grid canvas {tile_h * args.rows}x{tile_w * args.cols} "
              f"({canvas_area / 1024**2:.2f} MPix) is well beyond the ~1024x1024 the model "
              f"was trained around -- expect degraded results. Reduce --tile_size.")
    n = args.rows * args.cols
    frames = load_frames(args.frames_dir, n, uniform_sample=args.uniform_sample)
    grid_image = make_grid(frames, args.rows, args.cols, tile_w, tile_h)
    grid_image.save(os.path.join(args.output_dir, "input_grid.png"))

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=dtype, device=device,
        model_configs=[
            ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
        processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"))

    if args.mode == "image_grid":
        # Whole grid through the unmodified pipeline: one big image edit.
        # edit_image_auto_resize=False is ESSENTIAL: the pipeline otherwise
        # resizes the condition to 1024x1024 area, and (RoPE coords being
        # canvas-centered) its tokens then only cover the CENTRAL region of a
        # larger noise canvas -- the model edits the center and hallucinates a
        # blurry ring outside the condition's PE extent.
        out = pipe(args.prompt, negative_prompt=args.negative_prompt, cfg_scale=args.cfg_scale,
                   edit_image=grid_image,
                   height=tile_h * args.rows, width=tile_w * args.cols,
                   edit_image_auto_resize=False,
                   num_inference_steps=args.num_inference_steps, seed=args.seed,
                   zero_cond_t=args.zero_cond_t)
        out.save(os.path.join(args.output_dir, "edited_grid.png"))
        print(f"[empirical] wrote {args.output_dir}/edited_grid.png")
        return

    # latent_grid: per-frame VAE encode, grid PE via the RoPE patch, then the
    # denoising loop and per-frame decode -- hand-rolled to show every step.
    from model import model_fn_video_tokens
    from projections import WanToQwenProjection, QwenToWanProjection

    def encode_frame(img):
        x = torch.from_numpy(np.asarray(img.resize((tile_w, tile_h)), dtype=np.float32) / 127.5 - 1.0)
        x = x.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
        return pipe.vae.encode(x)

    latents = torch.stack([encode_frame(fr)[0] for fr in frames], dim=1).unsqueeze(0)  # (1,16,N,h,w)

    # Identity projections initialized from img_in/proj_out: the DiT sees
    # per-frame latents embedded exactly as it embeds its own image latents.
    in_proj = WanToQwenProjection(16, pipe.dit.img_in.out_features)
    out_proj = QwenToWanProjection(16, pipe.dit.img_in.out_features)
    in_proj.init_from_qwen_dit(pipe.dit)
    out_proj.init_from_qwen_dit(pipe.dit)
    in_proj.to(device=device, dtype=dtype)
    out_proj.to(device=device, dtype=dtype)

    emb = QwenImageUnit_PromptEmbedder().process(pipe, prompt=args.prompt, edit_image=grid_image)
    use_cfg = args.cfg_scale > 1.0
    if use_cfg:
        neg_emb = QwenImageUnit_PromptEmbedder().process(
            pipe, prompt=args.negative_prompt, edit_image=grid_image)
    noise_seq_len = latents.shape[2] * (latents.shape[3] // 2) * (latents.shape[4] // 2)
    pipe.scheduler.set_timesteps(args.num_inference_steps, dynamic_shift_len=noise_seq_len)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    x = torch.randn(latents.shape, generator=gen).to(device=device, dtype=dtype)

    def forward(e, t):
        return model_fn_video_tokens(
            pipe.dit, in_proj, out_proj, latents=x, ref_latents=latents,
            prompt_emb=e["prompt_emb"], prompt_emb_mask=e["prompt_emb_mask"],
            timestep=t, latent_grid=(args.rows, args.cols),
            zero_cond_t=args.zero_cond_t, pe_mode="grid")

    for pid, t in enumerate(pipe.scheduler.timesteps):
        t = t.unsqueeze(0).to(dtype=dtype, device=device)
        pred = forward(emb, t)
        if use_cfg:
            # True CFG with norm-preserving rescale (per spatial position over
            # the channel dim), matching the diffusers-based reference script.
            pred_neg = forward(neg_emb, t)
            comb = pred_neg + args.cfg_scale * (pred - pred_neg)
            cond_norm = torch.norm(pred, dim=1, keepdim=True)
            comb_norm = torch.norm(comb, dim=1, keepdim=True)
            pred = comb * (cond_norm / comb_norm)
        x = pipe.scheduler.step(pred, pipe.scheduler.timesteps[pid], x)

    for i in range(n):
        img = pipe.vae.decode(x[0, :, i].unsqueeze(0))
        img = pipe.vae_output_to_image(img)
        img.save(os.path.join(args.output_dir, f"edited_frame_{i:02d}.png"))
    print(f"[empirical] wrote {n} edited frames to {args.output_dir}/")


if __name__ == "__main__":
    main()
