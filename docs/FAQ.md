# ❓ FAQ (video-enhance)

## Q1. "CUDA not available" error
PyTorch CPU wala install hai. CUDA wala install karo:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

## Q2. Model load pe "storage has wrong byte size" / "archive is corrupted"
Model file corrupt/download adhura hai. Dobara download karo:
```bash
python scripts/download_models.py
```

## Q3. NVENC "No capable devices found"
AI modes me x264 (CPU) auto-fallback hota hai — koi action nahi.
Turbo/best me NVENC use hota hai jab available ho.

## Q4. Bahut slow hai
- `fastai` mode use karo (2x, pre-down)
- chhota model: `realesr-general-x4v3.pth` (4.7MB) sabse fast
- `hq` sirf tab jab quality zaroori ho (4x full)

## Q5. GPU garam ho raha hai (>87C)
Auto-pause laga hai (thermal guard). Fan clean karo, laptop par cooling pad use karo, ya `fastai` chalao.

## Q6. Output video me audio nahi?
ffmpeg `-map 1:a?` audio pipe karta hai — agar source me audio ho to output me bhi aana chahiye. Kuch codecs (e.g. webm/VP9) ka audio output me nahi aa sakta — pehle `ffmpeg -i video.mp4` se container check karo.

## Q7. 4K output chahiye, lekin 2x hi mil raha hai
`fastai` = 2x (full 4K ke liye 1080p → 2160p maana jata hai, correct).
`hq` = 4x full (1080p → 4320p). Dhyan: 4x ka output bahut bada hota hai.

## Q8. Kaggle/Colab model kyu use karu?
Local GPU chhoti hai (4GB) → slow. FlashVSR+ (cloud, free) diffusion-grade quality deta hai — `notebooks/flashvsr_kaggle.ipynb` kholo.

## Q9. "No module named 'av'" / 'spandrel'
`pip install -r requirements.txt` chalao (venv me).

## Q10. Progress file kaha hai?
`output/enhanced/_progress.txt` — GUI isi se % + ETA dikhata hai.
