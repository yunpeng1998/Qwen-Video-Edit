"""Repo root doubles as a ComfyUI custom-node package: clone this repo into
ComfyUI/custom_nodes/ and the nodes in comfyui_nodes/ are picked up. Outside
ComfyUI this file is inert -- the guard below keeps normal usage
(train.py / infer.py / empirical.py) unaffected."""

try:
    from .comfyui_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # noqa: BLE001 -- missing deps outside ComfyUI are fine
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}
    print(f"[qwen-video-edit] ComfyUI nodes not loaded: {_e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
