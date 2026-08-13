"""Regenerate the project page's gallery manifest.

Drop your curated results into assets/gallery/ as triples following the
inference pipeline's own output convention:

    assets/gallery/{name}_enhanced_f4b4.mp4    # edited result
    assets/gallery/{name}_input.mp4            # original video
    assets/gallery/{name}.txt                  # the edit prompt

then run `python build_gallery.py` -- it writes assets/gallery.js, which
index.html loads. Entries missing their _input.mp4 are skipped with a
warning; a missing .txt just leaves the prompt empty.
"""
import json
import os
import subprocess

GALLERY_DIR = os.path.join("assets", "gallery")
OUT = os.path.join("assets", "gallery.js")
RESULT_SUFFIX = "_enhanced_f4b4"


def video_dims(path: str):
    """(width, height) via ffprobe; None if unavailable."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
            text=True).strip()
        w, h = map(int, out.split(",")[:2])
        return w, h
    except Exception as e:  # noqa: BLE001 -- no ffprobe / unreadable file
        print(f"[gallery] warn: could not probe {path} ({e}); treating as landscape")
        return None

items = []
if os.path.isdir(GALLERY_DIR):
    for fname in sorted(os.listdir(GALLERY_DIR)):
        if not fname.endswith(f"{RESULT_SUFFIX}.mp4"):
            continue
        name = fname[: -len(f"{RESULT_SUFFIX}.mp4")]
        before = os.path.join(GALLERY_DIR, f"{name}_input.mp4")
        if not os.path.exists(before):
            print(f"[gallery] skip {name}: no {name}_input.mp4")
            continue
        prompt = ""
        txt = os.path.join(GALLERY_DIR, f"{name}.txt")
        if os.path.exists(txt):
            prompt = open(txt, encoding="utf-8").read().strip()
        dims = video_dims(os.path.join(GALLERY_DIR, fname))
        portrait = bool(dims and dims[1] > dims[0])
        items.append({
            "name": name,
            "before": f"assets/gallery/{name}_input.mp4",
            "after": f"assets/gallery/{fname}",
            "prompt": prompt,
            "portrait": portrait,
        })

# landscape first, portrait after; alphabetical within each group
items.sort(key=lambda it: (it["portrait"], it["name"]))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("window.GALLERY = " + json.dumps(items, ensure_ascii=False, indent=1) + ";\n")
print(f"[gallery] wrote {OUT} with {len(items)} items")
