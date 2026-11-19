#!/usr/bin/env python3
"""GUI for video-enhance. Run: python enhance_gui.py"""

import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
SCRIPTS = HERE
OUT_DIR = os.environ.get("ENHANCE_OUT", r"D:\ComfyUI\output\enhanced")
MODELS_DIR = os.environ.get(
    "ENHANCE_MODELS",
    r"D:\ComfyUI\ComfyUI (1)\ComfyUI\models\upscale_models")
PROGRESS_TXT = os.path.join(OUT_DIR, "_progress.txt")

MODES = ["turbo (FAST - filters only)",
         "best (HQ filters + denoise)",
         "fastai (AI 2x fast - recommended)",
         "hq (AI 4K full - slow, max quality)",
         "night denoise (AI 2x/4x)"]
SIGMA_MAP = {"Low (10)": 10.0, "Medium (20)": 20.0, "High (30)": 30.0}
PROG_RE = re.compile(r"\[(\d+)/(\d+)\]")


def find_models():
    out = []
    if os.path.isdir(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.lower().endswith(".pth") and not f.startswith("put_"):
                out.append(f)
    if not out:
        out = ["realesr-general-x4v3.pth"]
    return sorted(out)


class App:
    def __init__(self, root):
        self.root = root
        root.title("Video Enhancer - 4K AI")
        root.geometry("760x640")
        self.q = queue.Queue()
        self.proc = None
        self.running = False
        pad = {"padx": 8, "pady": 4}

        frm = ttk.Frame(root)
        frm.pack(fill="x", padx=10, pady=8)
        ttk.Label(frm, text="Video file:").grid(row=0, column=0,
                                                sticky="w", **pad)
        self.video_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.video_var).grid(
            row=0, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Browse...", command=self.browse).grid(
            row=0, column=2, **pad)

        ttk.Label(frm, text="Mode:").grid(row=1, column=0, sticky="w", **pad)
        self.mode_var = tk.StringVar(value=MODES[2])
        ttk.Combobox(frm, textvariable=self.mode_var, values=MODES,
                     state="readonly", width=34).grid(
            row=1, column=1, sticky="w", **pad)

        ttk.Label(frm, text="AI Model:").grid(row=2, column=0,
                                              sticky="w", **pad)
        self.model_var = tk.StringVar(value=find_models()[3 % len(find_models())])
        ttk.Combobox(frm, textvariable=self.model_var, values=find_models(),
                     state="readonly", width=34).grid(
            row=2, column=1, sticky="w", **pad)

        ttk.Label(frm, text="Noise reject:").grid(row=3, column=0,
                                                  sticky="w", **pad)
        self.noise_var = tk.IntVar(value=25)
        ttk.Scale(frm, from_=10, to=35, variable=self.noise_var,
                  orient="horizontal", length=220).grid(
            row=3, column=1, sticky="w", **pad)
        ttk.Label(frm, textvariable=self.noise_var).grid(
            row=3, column=2, sticky="w", **pad)

        ttk.Label(frm, text="Night sigma:").grid(row=4, column=0,
                                                 sticky="w", **pad)
        self.sigma_var = tk.StringVar(value="Medium (20)")
        ttk.Combobox(frm, textvariable=self.sigma_var,
                     values=list(SIGMA_MAP), state="readonly",
                     width=16).grid(row=4, column=1, sticky="w", **pad)
        ttk.Label(frm, text="Night upscale:").grid(
            row=4, column=1, sticky="e", **pad)
        self.nup_var = tk.StringVar(value="4x")
        ttk.Combobox(frm, textvariable=self.nup_var,
                     values=["off", "2x", "4x"], state="readonly",
                     width=6).grid(row=4, column=2, sticky="w", **pad)

        btns = ttk.Frame(root)
        btns.pack(fill="x", padx=10, pady=4)
        self.run_btn = ttk.Button(btns, text="START", command=self.start)
        self.run_btn.pack(side="left", padx=4)
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self.cancel,
                                     state="disabled")
        self.cancel_btn.pack(side="left", padx=4)
        ttk.Button(btns, text="Open Output Folder",
                   command=self.open_out).pack(side="right", padx=4)
        ttk.Button(btns, text="Check GPU Temp",
                   command=self.gpu_temp).pack(side="right", padx=4)

        self.prog = ttk.Progressbar(root, mode="determinate")
        self.prog.pack(fill="x", padx=10, pady=4)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(root, textvariable=self.status_var).pack(fill="x", padx=10)

        logf = ttk.Frame(root)
        logf.pack(fill="both", expand=True, padx=10, pady=6)
        self.txt = tk.Text(logf, height=18, bg="#0d1117", fg="#c9d1d9",
                           insertbackground="white", font=("Consolas", 9))
        sb = ttk.Scrollbar(logf, command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt.pack(side="left", fill="both", expand=True)
        self.txt.tag_configure("done", foreground="#3fb950")
        self.txt.tag_configure("err", foreground="#f85149")

        root.after(100, self.poll)
        self.log("Ready. Video select karo, mode chuno, START dabao.\n")

    def browse(self):
        p = filedialog.askopenfilename(
            title="Video select karo",
            filetypes=[("Video", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v"),
                       ("All", "*.*")])
        if p:
            self.video_var.set(p)

    def open_out(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        os.startfile(OUT_DIR)

    def gpu_temp(self):
        try:
            o = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu",
                 "--format=csv,noheader,nounits"], text=True).strip()
            self.log("GPU Temp: %s C\n" % o)
        except Exception:
            self.log("nvidia-smi nahi mila\n")

    def log(self, msg, tag=None):
        self.txt.insert("end", msg, tag)
        self.txt.see("end")

    def build_cmd(self):
        video = self.video_var.get().strip()
        if not video or not os.path.isfile(video):
            messagebox.showerror("Error", "Pehle video file select karo!")
            return None, None
        mode = self.mode_var.get()
        noise = -abs(int(self.noise_var.get()))
        env = dict(os.environ)
        if mode.startswith("night"):
            up = self.nup_var.get()
            if up != "off":
                env["ENHANCE_MODEL"] = os.path.join(
                    MODELS_DIR, self.model_var.get())
            cmd = [PYTHON, os.path.join(SCRIPTS, "night_enhance.py"),
                   video, "--sigma",
                   str(int(SIGMA_MAP[self.sigma_var.get()])),
                   "--upscale", up]
            return cmd, env
        if mode.startswith("fastai") or mode.startswith("hq"):
            env["ENHANCE_MODEL"] = os.path.join(
                MODELS_DIR, self.model_var.get())
            m = "fastai" if mode.startswith("fastai") else "hq"
        elif mode.startswith("best"):
            m = "best"
        else:
            m = "turbo"
        cmd = [PYTHON, os.path.join(SCRIPTS, "enhance.py"), video, m,
               str(noise)]
        return cmd, env

    def start(self):
        if self.running:
            return
        cmd, env = self.build_cmd()
        if cmd is None:
            return
        self.running = True
        self.run_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.prog.config(value=0)
        self.status_var.set("starting...")
        self.log("\n=== START: " + " ".join(
            [os.path.basename(c) for c in cmd[1:3] + cmd[3:]]) + " ===\n")
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW)
        threading.Thread(target=self.reader, daemon=True).start()

    def reader(self):
        try:
            for line in self.proc.stdout:
                self.q.put(("line", line))
            self.proc.wait()
            self.q.put(("done", self.proc.returncode))
        except Exception:
            self.q.put(("done", -1))

    def cancel(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.log("\n[!] Cancel kiya gaya\n", "err")
        self.running = False
        self.cancel_btn.config(state="disabled")
        self.run_btn.config(state="normal")

    def poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "line":
                    self.handle_line(payload)
                else:
                    self.finish(payload)
        except queue.Empty:
            pass
        if self.running:
            self.read_progress_file()
        self.root.after(100, self.poll)

    def read_progress_file(self):
        try:
            with open(PROGRESS_TXT, "r", encoding="utf-8",
                      errors="replace") as f:
                line = f.read().strip()
            if not line:
                return
            self.status_var.set(line)
            m = re.search(r"(\d+)/(\d+)\s+([\d.]+)%", line)
            if m:
                self.prog.config(maximum=int(m.group(2)),
                                 value=int(m.group(1)))
        except Exception:
            pass

    def handle_line(self, line):
        line = line.rstrip("\n").rstrip("\r")
        if "\r" in line:
            line = line.split("\r")[-1]
        m = PROG_RE.search(line)
        if m:
            try:
                self.prog.config(maximum=int(m.group(2)),
                                 value=int(m.group(1)))
            except ValueError:
                pass
        tag = None
        if "DONE" in line or "!" in line[:6]:
            tag = "done"
        elif "ERROR" in line.upper() or "Error" in line:
            tag = "err"
        self.log(line + "\n", tag)

    def finish(self, code):
        self.running = False
        self.cancel_btn.config(state="disabled")
        self.run_btn.config(state="normal")
        self.proc = None
        if code == 0:
            self.status_var.set("DONE - output: " + OUT_DIR)
            self.log("\nFinished. Output: %s\n" % OUT_DIR, "done")
        else:
            self.status_var.set("FAILED exit=" + str(code))
            self.log("\nProcess exit code: %s (upar log check karo)\n"
                     % code, "err")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
