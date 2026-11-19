#!/usr/bin/env python3
"""
download_models.py - AI upscale models download karo (auto).

Usage:
  python scripts/download_models.py            # sab models (default)
  python scripts/download_models.py --only realesr-general-x4v3.pth
"""

import argparse
import os
import sys
import urllib.request

DEFAULT_DIR = os.environ.get(
    "ENHANCE_MODELS",
    r"D:\ComfyUI\ComfyUI (1)\ComfyUI\models\upscale_models")

MODELS = {
    "RealESRGAN_x2plus.pth": {
        "url": "https://huggingface.co/uwg/upscaler/resolve/main/Face_Restore/FaceFusion/real_esrgan_x2plus.pth",
        "size_mb": 64, "scale": 2},
    "realesr-general-x4v3.pth": {
        "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/realesr-general-x4v3.pth",
        "size_mb": 5, "scale": 4},
    "4x-UltraSharp.pth": {
        "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth",
        "size_mb": 64, "scale": 4},
    "4x_NMKD-Superscale-SP_178000_G.pth": {
        "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Superscale-SP_178000_G.pth",
        "size_mb": 64, "scale": 4},
}


def download(url, dst):
    print("downloading:", os.path.basename(dst))
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dst)


def verify(path, expected_mb):
    if not os.path.exists(path):
        return True
    mb = os.path.getsize(path) / 1048576.0
    return abs(mb - expected_mb) < 8  # allow tolerance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--only", help="sirf ye model file")
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    for name, meta in MODELS.items():
        if args.only and args.only != name:
            continue
        path = os.path.join(args.dir, name)
        if os.path.exists(path) and verify(path, meta["size_mb"]):
            print("✓ (already)", name, meta["scale"], "x")
            continue
        try:
            download(meta["url"], path)
            print("✓", name, "| scale", meta["scale"], "x | %.0f MB" %
                  (os.path.getsize(path) / 1048576.0))
        except Exception as e:
            print("✗", name, "FAIL:", e)
    print("done ->", args.dir)


if __name__ == "__main__":
    main()
