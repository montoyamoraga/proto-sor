#!/usr/bin/env python3
"""
Stitch a folder of frames into a fixed-framerate video with narration on
top: every distinct object YOLO finds in a frame is spoken all at once --
overlapping, in parallel -- rather than read out one after another, so a
frame with ten cars doesn't take ten times as long to narrate as a frame
with one. A class that appears more than once in a frame is pitched up,
so "several cars" sounds noticeably different from a single "car".

macOS only: narration is synthesized with the built-in `say` command.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from collections import Counter

import numpy as np

from coco_es import DEFAULT_VOICES, translate
from yolo_filter import load_model
from yolo_labels import detect_labels


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def require(cmd):
    if shutil.which(cmd) is None:
        raise SystemExit(f"required command not found: {cmd}")


def pitch_factor(count, step):
    if count <= 1:
        return 1.0
    return 1.0 + step * (count - 1)


def atempo_chain(target):
    """ffmpeg's atempo filter only accepts 0.5-2.0 per instance; chain
    several to reach tempo multipliers outside that range."""
    stages = []
    remaining = target
    if remaining > 2.0:
        while remaining > 2.0:
            stages.append(2.0)
            remaining /= 2.0
    elif remaining < 0.5:
        while remaining < 0.5:
            stages.append(0.5)
            remaining /= 0.5
    stages.append(remaining)
    return ",".join(f"atempo={s:.6f}" for s in stages)


def make_word_clip(word, factor, voice, rate, sr, workdir, tag):
    """Synthesize `word` and return it as an int16 mono numpy array at `sr`,
    sped up/pitched up by `factor` (chipmunk-style: higher factor = higher
    voice, same trick as asetrate on a sample player)."""
    raw_path = os.path.join(workdir, f"raw_{tag}.aiff")
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", raw_path, word], check=True, capture_output=True)

    wav_path = os.path.join(workdir, f"clip_{tag}.wav")
    if factor != 1.0:
        af = f"asetrate={sr}*{factor:.4f},aresample={sr},{atempo_chain(1.0 / factor)}"
        cmd = ["ffmpeg", "-y", "-i", raw_path, "-af", af, "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", wav_path]
    else:
        cmd = ["ffmpeg", "-y", "-i", raw_path, "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", wav_path]
    subprocess.run(cmd, check=True, capture_output=True)

    with wave.open(wav_path, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", default="frames_models_added_up", help="frames to show on screen")
    ap.add_argument("--labels-dir", default="", help="frames to run YOLO on for narration; defaults to --frames-dir")
    ap.add_argument("--frame-pattern", default="frame_%05d.jpg", help="printf-style pattern ffmpeg uses to read the frames")
    ap.add_argument("--pattern", default="frame_*.jpg", help="glob pattern used to count/list the frames")
    ap.add_argument("--output", default="out_narrated.mp4")
    ap.add_argument("--fps", type=float, default=5.0, help="playback framerate")
    ap.add_argument("--model", default="yolov8n.pt", help="a plain detection model is enough; masks aren't needed here")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", default="", help="comma-separated class names to keep (default: all)")
    ap.add_argument("--lang", default="en", choices=["en", "es"], help="narration language; translates COCO class names and picks a matching default voice")
    ap.add_argument("--voice", default="", help="run `say -v ?` to list installed voices (default: Samantha for en, Mónica for es)")
    ap.add_argument("--rate", type=int, default=400, help="words per minute; say's own default is ~175-200")
    ap.add_argument("--sample-rate", type=int, default=22050)
    ap.add_argument("--pitch-step", type=float, default=0.05, help="pitch/speed multiplier added per extra duplicate of a class within one frame; uncapped")
    args = ap.parse_args()

    require("say")
    require("ffmpeg")

    voice = args.voice or DEFAULT_VOICES[args.lang]
    sr = args.sample_rate

    labels_dir = args.labels_dir or args.frames_dir
    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, args.pattern)))
    label_paths = sorted(glob.glob(os.path.join(labels_dir, args.pattern)))
    if not frame_paths:
        raise SystemExit(f"no frames matching {args.pattern!r} in {args.frames_dir}/")
    if len(frame_paths) != len(label_paths):
        raise SystemExit(
            f"{args.frames_dir}/ has {len(frame_paths)} frames but {labels_dir}/ has {len(label_paths)} -- "
            "they need to correspond 1:1, frame by frame. Regenerate one to match the other."
        )
    log(f"{len(frame_paths)} frames at {args.fps}fps; detecting objects in {labels_dir}/ for narration...")

    model = load_model(args.model)
    class_ids = None
    if args.classes:
        wanted = {n.strip().lower() for n in args.classes.split(",") if n.strip()}
        name_to_id = {name.lower(): idx for idx, name in model.names.items()}
        unknown = wanted - set(name_to_id)
        if unknown:
            raise SystemExit(f"unknown class name(s): {', '.join(sorted(unknown))}")
        class_ids = [name_to_id[n] for n in wanted]

    workdir = tempfile.mkdtemp(prefix="narrate_")
    try:
        clip_cache = {}
        placements = []  # (start_sample, samples)
        for i, label_path in enumerate(label_paths):
            labels_en = detect_labels(model, label_path, args.conf, class_ids, "", "left-to-right")
            counts = Counter(labels_en)
            start_sample = round((i / args.fps) * sr)

            spoken = []
            for cls, n in counts.items():
                factor = pitch_factor(n, args.pitch_step)
                key = (cls, round(factor, 3))
                if key not in clip_cache:
                    word = translate(cls, args.lang)
                    clip_cache[key] = make_word_clip(word, factor, voice, args.rate, sr, workdir, len(clip_cache))
                placements.append((start_sample, clip_cache[key]))
                spoken.append(f"{cls}x{n}" if n > 1 else cls)
            log(f"[{i}] {', '.join(spoken) if spoken else '(nothing)'}")

        video_samples = round(len(frame_paths) / args.fps * sr)
        total_samples = max([s + len(a) for s, a in placements] + [video_samples])
        master = np.zeros(total_samples, dtype=np.int64)
        for start, samples in placements:
            master[start:start + len(samples)] += samples.astype(np.int64)

        peak = int(np.max(np.abs(master))) if placements else 0
        if peak > 32767:
            master = master * (32767 * 0.98 / peak)
        master = np.clip(master, -32768, 32767).astype(np.int16)

        narration_wav = os.path.join(workdir, "narration.wav")
        with wave.open(narration_wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(master.tobytes())

        silent_video = os.path.join(workdir, "video.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", os.path.join(args.frames_dir, args.frame_pattern),
             "-pix_fmt", "yuv420p", "-c:v", "libx264", silent_video],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", silent_video, "-i", narration_wav,
             "-c:v", "copy", "-c:a", "aac", args.output],
            check=True,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    log(f"done -> {args.output}")


if __name__ == "__main__":
    main()
