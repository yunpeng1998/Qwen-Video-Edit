"""Single-node multi-GPU training (no Ray). Launch with torchrun:

    torchrun --nproc_per_node=8 train.py \
        --meta_paths /data/ditto/metadata/training_metadata/global.json \
        --video_root /data/ditto/videos \
        --num_frames 45 --video_max_pixels 245760 \
        --lora_base_model dit --lora_rank 32 \
        --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
        --use_gradient_checkpointing --zero_cond_t \
        --output_path ./checkpoints --save_steps 1000

Full fine-tuning instead of LoRA: replace the --lora_* flags with
--trainable_models dit (DeepSpeed ZeRO-2 + CPU optimizer offload engages
automatically; add --zero_stage 3 to also shard parameters).
Checkpoints are plain .safetensors written to --output_path, containing the
trainable weights only ("pipe.dit.*" full/LoRA + "in_proj.*"/"out_proj.*").
"""

import argparse
import os
from datetime import timedelta

import accelerate
import torch
from accelerate.utils import DeepSpeedPlugin

from diffsynth.diffusion import launch_training_task
from diffsynth.diffusion.logger import ModelLogger
from diffsynth.diffusion.parsers import (
    add_gradient_config, add_lora_config, add_offload_training_config,
    add_output_config, add_training_config,
)

from dataset import VideoEditDataset
from model import VideoTokenEditModule


def build_parser():
    parser = argparse.ArgumentParser(description="Train Qwen-Image-Edit on Wan video tokens.")
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_lora_config(parser)
    parser = add_gradient_config(parser)
    parser = add_offload_training_config(parser)

    parser.add_argument("--meta_paths", type=str, nargs="+", required=True,
                        help="Ditto-style meta JSON file(s).")
    parser.add_argument("--video_root", type=str, required=True,
                        help="Directory the meta files' relative video paths resolve against.")
    parser.add_argument("--num_frames", type=int, default=45)
    parser.add_argument("--video_height", type=int, default=384)
    parser.add_argument("--video_width", type=int, default=640)
    parser.add_argument("--video_max_pixels", type=int, default=0,
                        help="If > 0: per-video aspect-preserving sizing (overrides height/width).")
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--dataset_num_workers", type=int, default=8)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--model_id_with_origin_paths", type=str,
                        default="Qwen/Qwen-Image-Edit:transformer/diffusion_pytorch_model*.safetensors,"
                                "Qwen/Qwen-Image:text_encoder/model*.safetensors")
    parser.add_argument("--wan_vae_model_id_with_origin_path", type=str,
                        default="Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--latent_mode", type=str, default="wan_compressed",
                        choices=["wan_compressed", "qwen_framewise_pack4"])
    parser.add_argument("--pe_mode", type=str, default="grid", choices=["grid", "video"])
    parser.add_argument("--zero_cond_t", default=False, action="store_true")
    parser.add_argument("--no_init_projections_from_dit", default=False, action="store_true")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[2, 3])
    return parser


def main(args):
    dataset = VideoEditDataset(
        meta_paths=args.meta_paths, video_root=args.video_root,
        num_frames=args.num_frames, height=args.video_height, width=args.video_width,
        max_pixels=args.video_max_pixels or None,
        repeat=args.dataset_repeat, max_items=args.max_items or None,
    )

    # Build the model BEFORE the Accelerator: a stage-3 DeepSpeedPlugin
    # registers a global config that makes DiffSynth's loader construct
    # models under zero.Init (1-D sharded weights), breaking the projection
    # warm-start weight copies.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = VideoTokenEditModule(
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        wan_vae_model_id_with_origin_path=args.wan_vae_model_id_with_origin_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model, lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank, lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_from_checkpoint=args.resume_from_checkpoint,
        num_frames=args.num_frames, latent_mode=args.latent_mode, pe_mode=args.pe_mode,
        zero_cond_t=args.zero_cond_t,
        init_projections_from_dit=not args.no_init_projections_from_dit,
        device=f"cuda:{local_rank}",
    )

    kwargs_handlers = [accelerate.InitProcessGroupKwargs(timeout=timedelta(minutes=30))]
    deepspeed_plugin = None
    if args.trainable_models or args.zero_stage == 3:
        deepspeed_plugin = DeepSpeedPlugin(
            zero_stage=args.zero_stage,
            offload_optimizer_device="cpu" if args.trainable_models else None,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            zero3_save_16bit_model=args.zero_stage == 3,
            zero3_init_flag=False,
        )
    else:
        kwargs_handlers.append(
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters))

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=kwargs_handlers,
        mixed_precision="bf16",
    )

    # Plain local checkpointing; remove_prefix stays None on purpose --
    # infer.py dispatches on the "pipe.dit."/"in_proj."/"out_proj." prefixes.
    model_logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=None)
    launch_training_task(accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main(build_parser().parse_args())
