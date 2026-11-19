# 🎬 video-enhance

Free, local AI video enhancement & upscaling: denoise, color, sharpen, and
upscale videos up to 4K. No subscription, no cloud — runs on your own GPU.

Includes:
- **CLI** (`scripts/enhance.py`) + **GUI** (`scripts/enhance_gui.py`)
- **Night/low-light denoise** pipeline (`scripts/night_enhance.py`)
- **FlashVSR+ Kaggle notebook** (one-step diffusion upscaler, best quality)
- ESCGAN models auto-detected from `models/upscale_models/`

---

## ✨ Features

| Mode | What it does | Speed (GTX 1650) |
|---|---|---|
| `turbo` | ffmpeg filters: denoise + sharpen + color | real-time |
| `best`  | HQ filters: atadenoise + curves + vibrance | real-time |
| `fastai` | AI upscale 2x (fast, pre-downscaled) | ~1 fps @720p |
| `hq` | AI upscale 4x (max quality, slow) | ~0.02 fps @720p |
| `night` | FastDVDnet AI denoise + upscale 2x/4x | slow, for low-light |

AI upscale models supported (put `.pth` files in `upscale_models/`):
- `RealESRGAN_x2plus.pth` (2x)
- `realesr-general-x4v3.pth` (4x, fast)
- `4x-UltraSharp.pth` / `4x_NMKD-Superscale-SP_178000_G.pth` (4x, sharp)

## 🚀 Quickstart

```bash
# 1. install deps (CUDA pytorch recommended)
pip install -r requirements.txt

# 2. run CLI
python scripts/enhance.py video.mp4 fastai 25
#    set model:
set ENHANCE_MODEL=path\to\4x-UltraSharp.pth
python scripts/enhance.py video.mp4 hq 25

# 3. or launch GUI
python scripts/enhance_gui.py
```

Output goes to `output/enhanced/` (override with `ENHANCE_OUT`).
Live progress file: `output/enhanced/_progress.txt` (used by the GUI).

Environment overrides: `ENHANCE_FFMPEG`, `ENHANCE_OUT`, `ENHANCE_MODELS`,
`ENHANCE_MODEL`, `ENHANCE_FDN`.

## ☁️ Best quality: FlashVSR+ (cloud, free)

For diffusion-grade 4x upscaling (temporal consistency), open
[`notebooks/flashvsr_kaggle.ipynb`](notebooks/flashvsr_kaggle.ipynb) on
[Kaggle](https://www.kaggle.com) (GPU T4 x2) or Google Colab (T4).
Downloads ~7 GB of weights, then upscales uploaded videos to 1080p/1440p.

## 🧠 How it works

- AI modes: `spandrel` loads ESRGAN-style `.pth` models; frames are tiled
  (384px + overlap) for <4GB VRAM, upscaled on GPU, piped to ffmpeg.
- Encoder: NVENC when available (turbo/best); x264 on AI modes for stability.
- Thermal guard: pauses when GPU > 87°C.

## 📜 License

MIT — see [LICENSE](LICENSE). Upstream model weights have their own licenses
(ESRGAN: BSD-3; FlashVSR+: research license; use accordingly).
