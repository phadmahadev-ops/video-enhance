# 🐣 Beginner Guide (Hindi + English)

Ye guide unke liye hai jo pehli baar video enhancement use kar rahe hain.

## 1. Kya chahiye?

| Cheez | Zaroorat |
|---|---|
| Computer | Windows 10/11 (x64) |
| GPU | NVIDIA (CUDA) - GTX 1650 ya upar |
| Python | 3.10+ |
| Video | MP4/MKV/MOV |

## 2. Pehla step: Install

```bash
# Python packages
pip install -r requirements.txt
```

> PyTorch CUDA version ke liye: https://pytorch.org se apne CUDA ke hisaab se install karo.

## 3. Simple use (CLI)

```bash
# Sabse fast tarika (no AI, sirf filters):
python scripts/enhance.py video.mp4 turbo

# AI 2x upscale (recommended):
python scripts/enhance.py video.mp4 fastai 25

# AI 4x full quality (slow):
python scripts/enhance.py video.mp4 hq 25
```

Output: `output/enhanced/` folder me.

## 4. GUI use karna

```bash
python scripts/enhance_gui.py
```

GUI me:
1. **Browse** -> video select karo
2. **Mode** -> `fastai` (recommended) ya `hq`
3. **START** dabao
4. Progress bar + ETA dikhega, khatam hone par output folder khulega

## 5. Best quality - FlashVSR+ (cloud, free)

Local GPU slow hai to cloud use karo:

1. [Kaggle](https://www.kaggle.com) pe account banao (free)
2. `notebooks/flashvsr_kaggle.ipynb` file download karo
3. Kaggle → Code → New Notebook → **File → Import Notebook** → upload karo
4. Right side → **Settings → Accelerator → GPU T4 x2** + **Internet ON**
5. Run all cells → video upload karo → upscale ho jayega
6. `output.mp4` download karo

## 6. Common problems

| Problem | Solution |
|---|---|
| `CUDA not available` | PyTorch CUDA version install karo (s. 2) |
| `storage has wrong byte size` | Model file corrupt hai — dobara download karo |
| `No capable devices found` | Auto fallback x264 use hota hai — koi action nahi |
| Very slow (hq) | `fastai` mode use karo ya 2x model |
| GPU overheats | GUI me "Check GPU Temp" — fan clean karo |
