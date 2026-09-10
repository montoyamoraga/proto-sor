#!/usr/bin/env python3
"""
Run YOLO object detection/segmentation over a folder of frames and keep only
the pixels of the objects it finds, discarding the rest of the scene.

Inspired by Andreas Refsgaard's ml4a-style pieces that show computer vision
models' "point of view" -- only what the model recognizes remains visible,
everything else goes to the background color.

Uses Ultralytics YOLOv8. With a segmentation model (the "-seg" checkpoints,
e.g. yolov8n-seg.pt) each object is cut out along its actual silhouette;
with a plain detection model (yolov8n.pt) it falls back to keeping each
object's bounding-box rectangle.
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def clear_stale_frames(output_dir):
    """Remove any previously written frame_*.jpg/png so a shrunken input
    folder doesn't leave old frames behind past the new frame count."""
    for ext in ("jpg", "png"):
        for path in glob.glob(os.path.join(output_dir, f"frame_*.{ext}")):
            os.remove(path)


def load_model(name):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics is not installed. Run:\n"
            "  ./venv/bin/pip install -r requirements.txt"
        )
    return YOLO(name)


def background_canvas(size, mode, background):
    w, h = size
    if background == "transparent":
        return Image.new("RGBA", (w, h), (0, 0, 0, 0))
    color = {"black": (0, 0, 0), "white": (255, 255, 255)}[background]
    return Image.new(mode, (w, h), color)


def label_font(size=14):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def process_frame(model, in_path, out_path, args, font):
    image = Image.open(in_path).convert("RGB")
    np_image = np.array(image)

    results = model.predict(
        np_image,
        conf=args.conf,
        classes=args.class_ids,
        device=args.device or None,
        retina_masks=True,
        verbose=False,
    )[0]

    out_mode = "RGBA" if args.background == "transparent" else "RGB"
    canvas = background_canvas(image.size, out_mode, args.background)
    canvas_np = np.array(canvas)

    boxes = results.boxes
    masks = results.masks
    kept = 0

    if boxes is not None:
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].round().int().tolist()
            x1, y1, x2, y2 = xyxy
            if masks is not None:
                mask = masks.data[i].cpu().numpy() > 0.5
                if out_mode == "RGB":
                    canvas_np[mask] = np_image[mask]
                else:
                    canvas_np[mask, :3] = np_image[mask]
                    canvas_np[mask, 3] = 255
            else:
                if out_mode == "RGB":
                    canvas_np[y1:y2, x1:x2] = np_image[y1:y2, x1:x2]
                else:
                    canvas_np[y1:y2, x1:x2, :3] = np_image[y1:y2, x1:x2]
                    canvas_np[y1:y2, x1:x2, 3] = 255
            kept += 1

    out_image = Image.fromarray(canvas_np, out_mode)

    if args.labels and kept:
        draw = ImageDraw.Draw(out_image)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].round().int().tolist()
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            name = model.names[cls_id]
            text = f"{name} {conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            draw.text((x1 + 2, max(0, y1 - 16)), text, fill=(255, 0, 0), font=font)

    out_image.save(out_path)
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="frames")
    ap.add_argument("--output-dir", default="frames_yolo")
    ap.add_argument("--pattern", default="frame_*.jpg", help="glob pattern for input frames")
    ap.add_argument("--model", default="yolov8n-seg.pt", help="ultralytics model name or path; use a *-seg model for pixel-accurate cutouts")
    ap.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    ap.add_argument("--classes", default="", help="comma-separated class names to keep (default: all). e.g. 'car,person,bicycle'")
    ap.add_argument("--background", default="black", choices=["black", "white", "transparent"])
    ap.add_argument("--labels", action="store_true", help="draw class name + confidence + box outline over the cutouts")
    ap.add_argument("--device", default="", help="'cpu', 'mps', 'cuda:0', ... (default: let ultralytics choose)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    clear_stale_frames(args.output_dir)

    model = load_model(args.model)
    args.class_ids = None
    if args.classes:
        wanted = {n.strip().lower() for n in args.classes.split(",") if n.strip()}
        name_to_id = {name.lower(): idx for idx, name in model.names.items()}
        unknown = wanted - set(name_to_id)
        if unknown:
            raise SystemExit(f"unknown class name(s): {', '.join(sorted(unknown))}")
        args.class_ids = [name_to_id[n] for n in wanted]

    font = label_font()

    paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"no frames matching {args.pattern!r} in {args.input_dir}/")
    log(f"found {len(paths)} frames in {args.input_dir}/")

    ext = "png" if args.background == "transparent" else "jpg"
    total_kept = 0
    for i, in_path in enumerate(paths):
        out_path = os.path.join(args.output_dir, f"frame_{i:05d}.{ext}")
        kept = process_frame(model, in_path, out_path, args, font)
        total_kept += kept
        log(f"[{i}] {os.path.basename(in_path)}: kept {kept} object(s) -> {out_path}")

    log(f"done. wrote {len(paths)} frames to {args.output_dir}/ ({total_kept} objects total)")
    log(f"assemble into video with e.g.:")
    log(f"  ffmpeg -framerate 24 -i {args.output_dir}/frame_%05d.{ext} -c:v libx264 -pix_fmt yuv420p out_yolo.mp4")


if __name__ == "__main__":
    main()
