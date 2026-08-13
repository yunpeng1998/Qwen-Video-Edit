"""Ditto-style video-edit-pair dataset for open-source use: local video files
by default
--video_root at a local directory, see README).

Meta format (Ditto, https://github.com/EzioBy/Ditto): one or more JSON files,
each a list of {"source_path", "edited_path", "instruction"} entries with
paths relative to --video_root.
"""

import io
import json
import random

import numpy as np
import torch
from PIL import Image


def open_maybe_remote(path: str):
    if path.startswith("gs://"):
        try:
            import gcs  # internal-only, optional
            return gcs.open(path, "rb")
        except ImportError as e:
            raise RuntimeError(
                f"{path}: gs:// paths need an internal GCS reader that isn't "
                f"installed. Download the data locally and use local paths."
            ) from e
    return open(path, "rb")


def load_meta_entries(meta_paths: list[str]) -> list[dict]:
    entries = []
    for path in meta_paths:
        with open_maybe_remote(path) as f:
            data = json.load(f)
        assert isinstance(data, list), f"{path}: expected a JSON list"
        entries.extend(data)
        print(f"[dataset] Loaded {len(data)} entries from {path}.")
    return entries


def decode_video_frames(raw_bytes: bytes) -> list[np.ndarray]:
    """mp4 bytes -> list of uint8 HWC RGB frames (imageio + pyav)."""
    import imageio.v3 as iio
    frames = [np.asarray(fr) for fr in iio.imiter(io.BytesIO(raw_bytes), plugin="pyav", extension=".mp4")]
    if not frames:
        raise ValueError("decoded 0 frames")
    return frames


def adaptive_dims(frame_width: int, frame_height: int, max_pixels: int) -> tuple[int, int]:
    """Aspect-preserving (width, height): downscale to <= max_pixels (never
    upscale), each side rounded to a multiple of 16."""
    scale = min(1.0, (max_pixels / (frame_width * frame_height)) ** 0.5)
    return max(round(frame_width * scale / 16), 1) * 16, max(round(frame_height * scale / 16), 1) * 16


def frames_to_tensor(frames: list[np.ndarray], width: int, height: int) -> torch.Tensor:
    """uint8 HWC frames -> (C, T, H, W) float tensor in [-1, 1]."""
    resized = []
    for f in frames:
        img = Image.fromarray(f)
        if img.size != (width, height):
            img = img.resize((width, height), Image.BILINEAR)
        resized.append(np.asarray(img, dtype=np.float32))
    video = np.stack(resized) / 127.5 - 1.0
    return torch.from_numpy(video).permute(3, 0, 1, 2).contiguous()


def build_preview_grid(frames: list[np.ndarray], rows: int = 3, cols: int = 3,
                       target_area: int = 1024 * 1024) -> Image.Image:
    """Uniformly sampled frames tiled into one grid image -- the Qwen2.5-VL
    image prompt (the VL branch sees the whole video-as-grid)."""
    idx = np.linspace(0, len(frames) - 1, rows * cols).round().astype(int)
    tiles = [Image.fromarray(frames[i]) for i in idx]
    w0, h0 = tiles[0].size
    scale = (target_area / (w0 * cols * h0 * rows)) ** 0.5
    tw, th = max(int(w0 * scale) // 2 * 2, 2), max(int(h0 * scale) // 2 * 2, 2)
    grid = Image.new("RGB", (tw * cols, th * rows))
    for i, tile in enumerate(tiles):
        grid.paste(tile.resize((tw, th), Image.BILINEAR), ((i % cols) * tw, (i // cols) * th))
    return grid


def first_n_frames_padded(frames: list, n: int) -> list:
    return frames[:n] if len(frames) >= n else frames + [frames[-1]] * (n - len(frames))


class VideoEditDataset(torch.utils.data.Dataset):
    load_from_cache = False  # required by diffsynth's training runner

    def __init__(self, meta_paths, video_root, num_frames=45, height=384, width=640,
                 max_pixels=None, repeat=1, max_items=None, seed=0):
        assert (num_frames - 1) % 4 == 0 or num_frames % 4 == 0, \
            "num_frames must be 4k+1 (wan_compressed) or 4k (qwen_framewise_pack4)"
        self.entries = load_meta_entries(meta_paths)
        if max_items:
            rng = random.Random(seed)
            rng.shuffle(self.entries)
            self.entries = self.entries[:max_items]
        self.video_root = video_root.rstrip("/")
        self.num_frames, self.height, self.width = num_frames, height, width
        self.max_pixels = max_pixels
        self.repeat = repeat
        self._duplicate_first_component = None  # ditto GCS layout quirk (internal)

    def __len__(self):
        return len(self.entries) * self.repeat

    def _fetch_video_bytes(self, rel_path: str) -> bytes:
        plain = f"{self.video_root}/{rel_path}"
        duplicated = f"{self.video_root}/{rel_path.split('/')[0]}/{rel_path}"
        order = [duplicated, plain] if self._duplicate_first_component else [plain, duplicated]
        last_err = None
        for path in order:
            try:
                with open_maybe_remote(path) as f:
                    data = f.read()
                self._duplicate_first_component = path.count(rel_path.split("/")[0]) > 1
                return data
            except Exception as e:  # noqa: BLE001 -- MUST be broad: remote
                # backends raise their own not-found types (e.g. GCS's
                # google.api_core.exceptions.NotFound is NOT an OSError);
                # anything narrower silently skips the second path layout.
                last_err = e
        raise FileNotFoundError(f"Could not fetch {rel_path} under {self.video_root}: {last_err}")

    def _load_item(self, entry):
        src = decode_video_frames(self._fetch_video_bytes(entry["source_path"]))
        tgt = decode_video_frames(self._fetch_video_bytes(entry["edited_path"]))
        usable = min(len(src), len(tgt))
        if usable >= self.num_frames:
            start = random.randint(0, usable - self.num_frames)
            src_win, tgt_win = src[start:start + self.num_frames], tgt[start:start + self.num_frames]
        else:
            pad = self.num_frames - usable
            src_win = src[:usable] + [src[usable - 1]] * pad
            tgt_win = tgt[:usable] + [tgt[usable - 1]] * pad

        if self.max_pixels:
            h0, w0 = src_win[0].shape[:2]
            width, height = adaptive_dims(w0, h0, self.max_pixels)
        else:
            width, height = self.width, self.height

        return {
            "source_video": frames_to_tensor(src_win, width, height),
            "target_video": frames_to_tensor(tgt_win, width, height),
            "prompt": entry["instruction"],
            "preview_image": build_preview_grid(src_win),
        }

    def __getitem__(self, index):
        base = index % len(self.entries)
        for attempt in range(10):
            try:
                return self._load_item(self.entries[base])
            except Exception as e:  # noqa: BLE001 -- bad videos happen at scale
                print(f"[dataset] Failed entry {base}: {e} -- retrying ({attempt + 1}/10).")
                base = random.randint(0, len(self.entries) - 1)
        raise RuntimeError("10 consecutive dataset entries failed to load.")
