"""Compatibility shim. The grid/video RoPE support is now BAKED INTO the
vendored `diffsynth/models/qwen_image_dit.py` (this repo ships a pinned copy
of DiffSynth-Studio under ./diffsynth), so there is nothing to monkeypatch.
`apply()` just verifies the vendored patch is actually the one being
imported -- a guard against accidentally shadowing the vendored package with
an unpatched site-packages installation.
"""

import inspect


def apply():
    from diffsynth.models import qwen_image_dit

    source_path = inspect.getfile(qwen_image_dit)
    with open(source_path, "r") as f:
        patched = "GRID-TILE PATCH" in f.read()
    assert patched, (
        f"The imported diffsynth ({source_path}) is NOT the vendored patched "
        f"copy. Make sure you run from the repo root (so ./diffsynth shadows "
        f"any pip-installed diffsynth), or uninstall the pip package."
    )
