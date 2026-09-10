#!/usr/bin/env python3
"""
Run YOLO over a folder of frames and write out, per frame, the list of
objects it sees -- meant to be fed to a text-to-speech engine that narrates
what the model detects as the frames go by.
"""
import argparse
import glob
import os
import sys

from coco_es import DEFAULT_EMPTY_PLACEHOLDERS, translate
from yolo_filter import load_model


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def detect_labels(model, path, conf, class_ids, device, order):
    results = model.predict(path, conf=conf, classes=class_ids, device=device or None, verbose=False)[0]
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return []

    items = []
    for i in range(len(boxes)):
        x1 = float(boxes.xyxy[i][0])
        cls_id = int(boxes.cls[i])
        conf_i = float(boxes.conf[i])
        items.append((x1, model.names[cls_id], conf_i))

    if order == "left-to-right":
        items.sort(key=lambda t: t[0])
    else:
        items.sort(key=lambda t: -t[2])
    return [name for _, name, _ in items]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="frames")
    ap.add_argument("--output", default="frames_yolo_labels.txt")
    ap.add_argument("--pattern", default="frame_*.jpg", help="glob pattern for input frames")
    ap.add_argument("--model", default="yolov8n.pt", help="a plain detection model is enough; masks aren't needed here")
    ap.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    ap.add_argument("--classes", default="", help="comma-separated class names to keep (default: all)")
    ap.add_argument("--device", default="", help="'cpu', 'mps', 'cuda:0', ... (default: let ultralytics choose)")
    ap.add_argument("--order", default="left-to-right", choices=["left-to-right", "confidence"],
                     help="order objects appear in each frame's line: scanning left-to-right, or most confident first")
    ap.add_argument("--format", default="labeled", choices=["labeled", "plain"],
                     help="'labeled' prefixes each line with the frame name; 'plain' writes just the object list, "
                          "one frame per line, ready to pipe straight into a TTS engine")
    ap.add_argument("--lang", default="en", choices=["en", "es"], help="translates COCO class names into this language")
    ap.add_argument("--empty-placeholder", default="", help="text to write for a frame with no detections (default: 'nothing' for en, 'nada' for es)")
    args = ap.parse_args()

    empty_placeholder = args.empty_placeholder or DEFAULT_EMPTY_PLACEHOLDERS[args.lang]
    model = load_model(args.model)
    class_ids = None
    if args.classes:
        wanted = {n.strip().lower() for n in args.classes.split(",") if n.strip()}
        name_to_id = {name.lower(): idx for idx, name in model.names.items()}
        unknown = wanted - set(name_to_id)
        if unknown:
            raise SystemExit(f"unknown class name(s): {', '.join(sorted(unknown))}")
        class_ids = [name_to_id[n] for n in wanted]

    paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"no frames matching {args.pattern!r} in {args.input_dir}/")
    log(f"found {len(paths)} frames in {args.input_dir}/")

    lines = []
    for i, path in enumerate(paths):
        labels = [translate(name, args.lang) for name in detect_labels(model, path, args.conf, class_ids, args.device, args.order)]
        text = ", ".join(labels) if labels else empty_placeholder
        if args.format == "labeled":
            lines.append(f"{os.path.splitext(os.path.basename(path))[0]}: {text}")
        else:
            lines.append(text)
        log(f"[{i}] {os.path.basename(path)}: {text}")

    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")

    log(f"done. wrote {len(lines)} lines to {args.output}")


if __name__ == "__main__":
    main()
