"""Trainable projection layers bridging Wan 2.1 video VAE latents and the
Qwen-Image-Edit DiT token space.

Design mirrors the two ends it connects:

- `WanToQwenProjection` (input side) mirrors Wan DiT's own
  `patch_embedding = nn.Conv3d(16, dim, kernel_size=(1,2,2), stride=(1,2,2))`
  (diffsynth/models/wan_video_dit.py) -- no temporal compression, 2x2 spatial
  patchify -- but projects into the Qwen DiT inner dim (3072) so its output
  can be fed straight to the transformer blocks (bypassing `dit.img_in`).

- `QwenToWanProjection` (output side) mirrors Wan DiT's `Head`
  (`nn.Linear(dim, out_dim * prod(patch_size))` + unpatchify) -- Qwen's own
  conditioned `dit.norm_out` plays the role of Head's modulated norm, so this
  module is just the final linear + unpatchify (a drop-in replacement for
  `dit.proj_out`).

Both can be initialized FROM the pretrained Qwen DiT's `img_in` / `proj_out`
weights (`init_from_qwen_dit`). Qwen latents and Wan latents are both
16-channel, per-channel-normalized VAE latents, and Qwen tokens are exactly
2x2-patchified 16-channel vectors in (C, P, Q) flatten order -- so with this
init the transformer initially "sees" a Wan latent frame embedded exactly the
way it would see a Qwen image latent, and training only has to learn the
distribution shift rather than a from-scratch embedding.
"""

import torch
import torch.nn as nn
from einops import rearrange


class WanToQwenProjection(nn.Module):
    """Wan video latent (B, 16, T, h, w) -> Qwen DiT tokens
    (B, T * h/2 * w/2, inner_dim), frame-major scan order (frame 0's raster
    tokens first, then frame 1's, ...) -- MUST match the img_shapes entry
    order built in model_fn_video_tokens."""

    def __init__(self, in_channels: int = 16, inner_dim: int = 3072):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))

    @torch.no_grad()
    def init_from_qwen_dit(self, dit):
        """Copy dit.img_in (Linear(64, inner_dim)) into the conv kernel.

        Qwen's token features are `rearrange(latent, "B C (H P) (W Q) ->
        B (H W) (C P Q)", P=2, Q=2)` -- flat feature index = c*4 + p*2 + q.
        Conv3d weight (out, in_c, kT=1, kH=p, kW=q) multiplies
        latent[c, t, 2h+p, 2w+q], so conv.weight[o, c, 0, p, q] must equal
        img_in.weight[o, c*4 + p*2 + q], i.e. a plain .view()."""
        w = dit.img_in.weight.data  # (inner_dim, 64)
        out_dim, in_feat = w.shape
        c = in_feat // 4
        self.proj.weight.data.copy_(w.view(out_dim, c, 2, 2).unsqueeze(2))
        self.proj.bias.data.copy_(dit.img_in.bias.data)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        x = self.proj(latents)                      # (B, D, T, h/2, w/2)
        return rearrange(x, "B D T H W -> B (T H W) D")


class QwenToWanProjection(nn.Module):
    """Qwen DiT output tokens (B, T * h/2 * w/2, inner_dim) -> Wan video
    latent (B, 16, T, h, w). Inverse scan order of WanToQwenProjection."""

    def __init__(self, out_channels: int = 16, inner_dim: int = 3072):
        super().__init__()
        self.out_channels = out_channels
        self.proj = nn.Linear(inner_dim, out_channels * 4)  # 4 = 2x2 spatial patch

    @torch.no_grad()
    def init_from_qwen_dit(self, dit):
        """dit.proj_out is Linear(inner_dim, 64) with the same (C, P, Q)
        output flatten order this module unpatchifies with -- identical
        shape, direct copy."""
        self.proj.weight.data.copy_(dit.proj_out.weight.data)
        self.proj.bias.data.copy_(dit.proj_out.bias.data)

    def forward(self, tokens: torch.Tensor, num_frames: int, tokens_h: int, tokens_w: int) -> torch.Tensor:
        x = self.proj(tokens)  # (B, T*H*W, C*4)
        return rearrange(
            x, "B (T H W) (C P Q) -> B C T (H P) (W Q)",
            T=num_frames, H=tokens_h, W=tokens_w, P=2, Q=2,
        )


class FramewisePack4InProjection(nn.Module):
    """qwen_framewise_pack4 mode: PER-FRAME image latents (Qwen's own VAE,
    16ch, image statistics -- no temporal entanglement), packed 4 frames per
    token along the FEATURE dim. For frame group [4k, 4k+3]:

        token_k(patch) = concat_t( W_t( patch_feat(frame_{4k+t}) ) ),
        W_t: Linear(64, inner_dim/4)

    init_from_qwen_dit slices dit.img_in's weight rows into the 4 slots, so a
    STATIC group (4 identical frames) yields EXACTLY img_in(frame): the model
    starts by seeing packed video tokens as ordinary Qwen image tokens, and
    motion enters purely as slot-to-slot differences (linearly accessible)."""

    group = 4

    def __init__(self, in_channels: int = 16, inner_dim: int = 3072):
        super().__init__()
        assert inner_dim % self.group == 0
        self.slots = nn.ModuleList(
            nn.Linear(in_channels * 4, inner_dim // self.group) for _ in range(self.group)
        )

    @torch.no_grad()
    def init_from_qwen_dit(self, dit):
        w, b = dit.img_in.weight.data, dit.img_in.bias.data  # (inner, 64), (inner,)
        block = w.shape[0] // self.group
        for t, slot in enumerate(self.slots):
            slot.weight.data.copy_(w[t * block:(t + 1) * block])
            slot.bias.data.copy_(b[t * block:(t + 1) * block])

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # latents: (B, C, T, H, W) per-frame latents, T divisible by 4.
        x = rearrange(
            latents, "B C (N G) (H P) (W Q) -> B N G (H W) (C P Q)",
            G=self.group, P=2, Q=2,
        )
        token = torch.cat([self.slots[t](x[:, :, t]) for t in range(self.group)], dim=-1)
        return rearrange(token, "B N L D -> B (N L) D")  # group-major, per-group raster


class FramewisePack4OutProjection(nn.Module):
    """Inverse of FramewisePack4InProjection: each of the 4 frames in a token
    group reads the WHOLE token (motion info is cross-slot), via its own
    Linear(inner_dim, 64). init_from_qwen_dit copies dit.proj_out into every
    slot, so at init every frame decodes to exactly Qwen's proj_out(token)."""

    group = 4

    def __init__(self, out_channels: int = 16, inner_dim: int = 3072):
        super().__init__()
        self.out_channels = out_channels
        self.slots = nn.ModuleList(
            nn.Linear(inner_dim, out_channels * 4) for _ in range(self.group)
        )

    @torch.no_grad()
    def init_from_qwen_dit(self, dit):
        for slot in self.slots:
            slot.weight.data.copy_(dit.proj_out.weight.data)
            slot.bias.data.copy_(dit.proj_out.bias.data)

    def forward(self, tokens: torch.Tensor, num_frames: int, tokens_h: int, tokens_w: int) -> torch.Tensor:
        # num_frames here = number of token GROUPS (model_fn's tile count).
        x = rearrange(tokens, "B (N H W) D -> B N (H W) D", N=num_frames, H=tokens_h, W=tokens_w)
        frames = torch.stack([self.slots[t](x) for t in range(self.group)], dim=2)  # (B, N, G, HW, C*4)
        return rearrange(
            frames, "B N G (H W) (C P Q) -> B C (N G) (H P) (W Q)",
            H=tokens_h, W=tokens_w, P=2, Q=2,
        )
