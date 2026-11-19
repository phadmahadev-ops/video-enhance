#!/usr/bin/env python3
"""
night_enhance.py - low-light video denoise + AI upscale (2x/4x).

Used by the GUI and CLI. Requires CUDA + FastDVDnet weights in FDN_DIR.

Usage:
  python night_enhance.py <video> [--sigma 10|20|30] [--upscale off|2x|4x]
"""

import os
import subprocess
import sys
import time

FFMPEG = os.environ.get(
    "ENHANCE_FFMPEG",
    r"D:\ComfyUI\tools\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe")
OUT_DIR = os.environ.get("ENHANCE_OUT", r"D:\ComfyUI\output\enhanced")
FDN_DIR = os.environ.get(
    "ENHANCE_FDN",
    r"D:\ComfyUI\tools\fastdvdnet")
FDN_WEIGHTS = os.path.join(FDN_DIR, "model.pth")

from enhance import (  # noqa: E402
    upscale_frame, load_model, PROGRESS_TXT,
    probe_nvenc, encode_video)

FDN_TEMP_FRAMES = 5


def log(m):
    print(m, flush=True)


def load_fdn(fp16=True):
    import torch
    from models import FastDVDnet
    model = FastDVDnet(num_input_frames=FDN_TEMP_FRAMES)
    ckpt = torch.load(FDN_WEIGHTS, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt
                          else ckpt)
    model.to("cuda")
    if fp16:
        model.half()
    model.eval()
    return model


def fdn_denoise_chunk(model, frames, sigma, fp16, pre_down=0.5):
    import torch
    import torch.nn.functional as F
    seq = torch.stack(frames, dim=0).unsqueeze(0)  # [1,T,3,H,W]
    if pre_down != 1.0:
        seq = F.interpolate(seq, scale_factor=pre_down, mode="bicubic",
                            align_corners=False, antialias=True)
    if fp16:
        seq = seq.half()
    noise = torch.randn_like(seq) * (sigma / 255.0)
    with torch.inference_mode():
        out = model(seq + noise)
    return out[0]


def night_boost(img):
    """Light color/contrast boost for night footage."""
    import torch
    img = img.clamp(0, 1)
    img = (img - 0.5) * 1.12 + 0.5
    gain = torch.tensor([1.0, 1.05, 1.1], device=img.device).view(3, 1, 1)
    img = img * gain
    return img.clamp(0, 1)


def gpu_temp():
    try:
        o = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"], text=True)
        return int(o.strip())
    except Exception:
        return 0


def thermal_wait(limit=87):
    t = gpu_temp()
    while t >= limit:
        log("    temp %dC - pause..." % t)
        time.sleep(20)
        t = gpu_temp()


def build_encode_cmd(W, H, fps, video, dst, face_safe, audio_denoise,
                     up_scale=1):
    if face_safe:
        vf = ("deflicker=size=3,"
              "unsharp=5:5:0.45:5:5:0.2,"
              "cas=strength=0.35")
    else:
        vf = ("deflicker=size=3,"
              "unsharp=5:5:0.8:5:5:0.35,"
              "cas=strength=0.5")
    af = "afftdn=nf=-25" if audio_denoise else "anull"
    log("[*] encoder: x264 (CPU) - AI mode stability")
    return [FFMPEG, "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", "%dx%d" % (W * up_scale, H * up_scale),
            "-r", "%.6f" % fps, "-i", "-",
            "-i", video,
            "-map", "0:v", "-map", "1:a?",
            "-vf", vf, "-af", af,
            "-c:v", "libx264", "-preset", "medium", "-crf", "19",
            "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", dst]


def run(video, dst, sigma, upscale, face_safe=True, audio_denoise=True):
    import av
    import torch

    fp16 = True
    try:
        fdn = load_fdn(fp16=True)
    except Exception:
        log("[!] fp16 fail, fp32 use kar rahe hain")
        fp16 = False
        fdn = load_fdn(fp16=False)

    up_model = None
    up_scale = 1
    if upscale in ("2x", "4x"):
        up_model = load_model(os.environ.get("ENHANCE_MODEL"))
        up_scale = int(up_model.scale)

    src = av.open(video)
    vs = src.streams.video[0]
    W, H = vs.codec_context.width, vs.codec_context.height
    fps = float(vs.average_rate)
    total = vs.frames or 0
    log("[+] %dx%d @ %.2ffps | %d frames | sigma=%.0f | upscale=%s" %
        (W, H, fps, total, sigma, upscale or "off"))

    ff_log = open(os.path.join(OUT_DIR, "_ffmpeg_err.log"), "w")
    ff = subprocess.Popen(build_encode_cmd(W, H, fps, video, dst,
                                           face_safe, audio_denoise,
                                           up_scale),
                          stdin=subprocess.PIPE, stderr=ff_log)

    CHUNK = 30
    n, t0, buf = 0, time.time(), []

    def flush(frames):
        with torch.inference_mode():
            den = fdn_denoise_chunk(fdn, frames, sigma, fp16)
            for i in range(den.shape[0]):
                img = night_boost(den[i])
                if up_model is not None:
                    up = upscale_frame(up_model, img, up_scale, 1.0)
                    out = up.float().mul_(255).byte()
                else:
                    out = img.mul_(255).byte()
                ff.stdin.write(out.permute(1, 2, 0).cpu().numpy().tobytes())
        thermal_wait()

    for frame in src.decode(vs):
        arr = frame.to_ndarray(format="rgb24")
        t = torch.from_numpy(arr.copy()).to("cuda")
        t = t.permute(2, 0, 1).float().div_(255.0)
        buf.append(t)
        if len(buf) >= CHUNK:
            flush(buf)
            n += len(buf)
            buf = []
            el = time.time() - t0
            sp = n / el if el else 0
            eta = (total - n) / sp if sp > 0 and total else 0
            pct = n * 100.0 / total if total else 0
            try:
                with open(PROGRESS_TXT, "w") as pf:
                    pf.write("%d/%d %.1f%% %.2f fps ETA %d:%02d\n" %
                             (n, total, pct, sp,
                              int(eta // 60), int(eta % 60)))
            except Exception:
                pass
            print("\r[%d/%d] %.1f%% %.1f fps | ETA %d:%02d   " %
                  (n, total, pct, sp, int(eta // 60), int(eta % 60)),
                  end="", flush=True)
    if buf:
        flush(buf)
        n += len(buf)
    ff.stdin.close()
    ff.wait()
    src.close()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--sigma", type=float, default=None,
                    help="noise: L=10 M=20 H=30")
    ap.add_argument("--upscale", choices=["off", "2x", "4x"], default=None)
    ap.add_argument("--no-face-safe", action="store_true")
    ap.add_argument("--no-audio-denoise", action="store_true")
    a = ap.parse_args()

    if not os.path.isfile(a.video):
        log("File nahi mili: %s" % a.video)
        sys.exit(1)

    sigma = a.sigma if a.sigma is not None else 20.0
    upscale = a.upscale if a.upscale is not None else "off"
    face = not a.no_face_safe
    adenoise = not a.no_audio_denoise

    base = os.path.splitext(os.path.basename(a.video))[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    dst = os.path.join(OUT_DIR, base + "_night.mp4")

    t0 = time.time()
    run(a.video, dst, sigma, upscale,
        face_safe=face, audio_denoise=adenoise)

    mins, secs = divmod(int(time.time() - t0), 60)
    sz = os.path.getsize(dst)
    sz_str = "%.1f MB" % (sz / 1048576.0) if sz > 1048576 else \
        "%.0f KB" % (sz / 1024.0)
    log("")
    log("=" * 44)
    log("DONE! Time: %dm %02ds | Size: %s" % (mins, secs, sz_str))
    log(dst)
    log("=" * 44)


if __name__ == "__main__":
    main()
