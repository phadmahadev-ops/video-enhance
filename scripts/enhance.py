#!/usr/bin/env python3
"""
video-enhance: AI video enhancement / upscaling pipeline.

Modes:
  turbo    - fast ffmpeg filters only (denoise + sharpen + color)
  best     - heavy ffmpeg filters (atadenoise + curves + vibrance)
  fastai   - AI upscale (2x fast, pre-downscaled)
  hq       - AI upscale (full 4x, slow, max quality)

Usage:
  python enhance.py <video> [turbo|best|fastai|hq] [noise 10-35]
  set ENHANCE_MODEL=path/to/model.pth   (for AI modes)
"""

import os
import subprocess
import sys
import time

# ---- configuration (override via env or edit below) ----
FFMPEG = os.environ.get(
    "ENHANCE_FFMPEG",
    r"D:\ComfyUI\tools\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe")
OUT_DIR = os.environ.get("ENHANCE_OUT", r"D:\ComfyUI\output\enhanced")
MODELS_DIR = os.environ.get(
    "ENHANCE_MODELS",
    r"D:\ComfyUI\ComfyUI (1)\ComfyUI\models\upscale_models")
DEFAULT_MODEL = os.environ.get(
    "ENHANCE_MODEL", os.path.join(MODELS_DIR, "realesr-general-x4v3.pth"))

PROGRESS_TXT = os.path.join(OUT_DIR, "_progress.txt")
TILE = 384
OVERLAP = 16
BATCH = 3

FILTERS_TURBO = ("hqdn3d=1.5:1.5:6:6,"
                 "unsharp=5:5:0.8:5:5:0.4,"
                 "cas=strength=0.45,"
                 "eq=contrast=1.04:saturation=1.06")

FILTERS_BEST = ("atadenoise=s=9,"
                "hqdn3d=3:2:10:10,"
                "curves=master='0/0.015 0.25/0.28 0.75/0.8 1/0.985',"
                "eq=contrast=1.05:gamma=1.04:saturation=1.12,"
                "vibrance=intensity=0.12,"
                "unsharp=5:5:0.7:5:5:0.3,"
                "cas=strength=0.4")


def log(m):
    print(m, flush=True)


def probe_ff(video):
    out = subprocess.check_output(
        [os.path.join(os.path.dirname(FFMPEG), "ffprobe.exe"),
         "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0", video], text=True).strip()
    w, h, r = out.splitlines()[0].strip().split(",")[:3]
    num, den = r.split("/")
    fps = float(num) / float(den)
    return int(w), int(h), fps


def probe_rotation(video):
    """Original video ka rotation angle (display matrix)"""
    try:
        out = subprocess.check_output(
            [os.path.join(os.path.dirname(FFMPEG), "ffprobe.exe"),
             "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream_side_data_list=rotation",
             "-of", "json", video], text=True)
        import json as _json
        d = _json.loads(out)
        for sd in d["streams"][0].get("side_data_list", []):
            if "rotation" in sd:
                return float(sd["rotation"])
    except Exception:
        pass
    return 0.0


def probe_nvenc():
    try:
        subprocess.check_call(
            [FFMPEG, "-y", "-v", "error",
             "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1",
             "-frames:v", "1", "-c:v", "h264_nvenc",
             "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=40)
        return True
    except Exception:
        return False


def encode_video(enc, preset, q):
    if enc:
        return ["-c:v", "h264_nvenc", "-preset", preset, "-cq", str(q)]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(q)]


def run_turbo(video, dst, noise=-25, filters=None):
    enc = probe_nvenc()
    subprocess.check_call(
        [FFMPEG, "-y", "-v", "error", "-i", video,
         "-vf", filters or FILTERS_TURBO,
         "-af", "afftdn=nf=%d" % noise,
         *encode_video(enc, "p5", 19),
         "-pix_fmt", "yuv420p", "-profile:v", "high",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", dst])


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


def load_model(path=None):
    import torch
    from spandrel import ImageModelDescriptor, ModelLoader
    m = ModelLoader().load_from_file(path or DEFAULT_MODEL)
    assert isinstance(m, ImageModelDescriptor)
    m.to("cuda").half().eval()
    return m


def upscale_frame(model, frame_t, scale, pre_down):
    import torch
    with torch.inference_mode():
        _, H, W = frame_t.shape
        work = frame_t
        if pre_down != 1.0:
            work = torch.nn.functional.interpolate(
                frame_t.unsqueeze(0), scale_factor=pre_down,
                mode="bicubic", align_corners=False, antialias=True
            ).squeeze(0).clamp_(0, 1)
        _, h, w = work.shape
        out = torch.zeros((3, h * scale, w * scale), dtype=torch.float16,
                          device="cuda")
        for y in range(0, h, TILE):
            for x in range(0, w, TILE):
                y0, x0 = max(0, y - OVERLAP), max(0, x - OVERLAP)
                y1, x1 = min(h, y + TILE + OVERLAP), min(w, x + TILE + OVERLAP)
                patch = work[:, y0:y1, x0:x1].unsqueeze(0)
                up = model(patch)
                oy, ox = (y - y0) * scale, (x - x0) * scale
                eh, ew = (min(y + TILE, h) - y) * scale, \
                    (min(x + TILE, w) - x) * scale
                out[:, y * scale:y * scale + eh, x * scale:x * scale + ew] = \
                    up[0, :, oy:oy + eh, ox:ox + ew]
        return out


def run_ai(video, dst, pre_down, noise=-25, model_path=None):
    import av
    import torch

    model = load_model(model_path)
    scale = int(model.scale)

    src = av.open(video)
    vs = src.streams.video[0]
    W, H = vs.codec_context.width, vs.codec_context.height
    fps = float(vs.average_rate)
    total = vs.frames or 0
    log("[+] %dx%d @ %.2ffps | %d frames | mode=%s" %
        (W, H, fps, total, "FAST-AI" if pre_down == 0.5 else "HQ"))

    rot = probe_rotation(video)
    rot_vf = {"-90": "transpose=1", "90": "transpose=2"}.get(
        str(int(rot)), "")
    out_w, out_h = int(W * pre_down * scale), int(H * pre_down * scale)
    log("[*] encoder: x264 (CPU) - AI mode stability | out %dx%d" %
        (out_w, out_h))

    ff = subprocess.Popen(
        [FFMPEG, "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
         "%dx%d" % (out_w, out_h),
         "-r", "%.6f" % fps, "-i", "-", "-i", video,
         "-map", "0:v", "-map", "1:a?",
         "-vf", rot_vf or "null",
         "-af", "afftdn=nf=%d" % noise,
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-profile:v", "high",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", dst],
        stdin=subprocess.PIPE,
        stderr=open(os.path.join(OUT_DIR, "_ffmpeg_err.log"), "w"))

    n, t0, batch = 0, time.time(), []

    def flush(frames):
        for fr in frames:
            arr = fr.to_ndarray(format="bgr24")
            t = torch.from_numpy(arr).to("cuda")
            t = t.permute(2, 0, 1).float().div_(255.0).half()
            up = upscale_frame(model, t, scale, pre_down)
            ff.stdin.write(up.float().mul_(255).byte()
                           .permute(1, 2, 0).cpu().numpy().tobytes())
        thermal_wait()

    for frame in src.decode(vs):
        batch.append(frame)
        if len(batch) >= BATCH:
            flush(batch)
            n += len(batch)
            batch = []
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
    if batch:
        flush(batch)
        n += len(batch)
    ff.stdin.close()
    ff.wait()
    src.close()


def main():
    if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
        log("Usage: enhance.py <video> [turbo|fastai|hq] [noise: 10-35]")
        sys.exit(1)
    video = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "turbo"
    noise = -25
    if len(sys.argv) > 3:
        try:
            noise = -abs(int(sys.argv[3]))
        except ValueError:
            pass

    base = os.path.splitext(os.path.basename(video))[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    dst = os.path.join(OUT_DIR, base + "_enhanced.mp4")

    W, H, fps = probe_ff(video)
    log("[*] Mode: %s | noise: %d | %dx%d @ %.2ffps" %
        (mode.upper(), noise, W, H, fps))

    try:
        with open(PROGRESS_TXT, "w") as pf:
            pf.write("0/0 0.0%% starting %s...\n" % mode)
    except Exception:
        pass

    t0 = time.time()
    if mode == "turbo":
        run_turbo(video, dst, noise)
    elif mode == "best":
        run_turbo(video, dst, noise, FILTERS_BEST)
    elif mode in ("fastai", "hq"):
        run_ai(video, dst, 0.5 if mode == "fastai" else 1.0, noise,
               os.environ.get("ENHANCE_MODEL"))
    else:
        log("Unknown mode: %s" % mode)
        sys.exit(1)

    mins, secs = divmod(int(time.time() - t0), 60)
    mb = os.path.getsize(dst) / 1048576.0
    try:
        with open(PROGRESS_TXT, "w") as pf:
            pf.write("%d/%d 100.0%% DONE in %dm %02ds\n" %
                     (1, 1, mins, secs))
    except Exception:
        pass
    log("")
    log("=" * 44)
    log("DONE! Time: %dm %02ds | Size: %.0f MB" % (mins, secs, mb))
    log(dst)
    log("=" * 44)


if __name__ == "__main__":
    main()
