#!/usr/bin/env python3
"""
Run semantic segmentation over a folder of frames and keep only the ground
plane -- road and sidewalk (cement) pixels -- discarding buildings, sky,
vehicles, people, everything else.

Companion to yolo_filter.py: that one keeps the discrete objects YOLO finds;
this one keeps the surface a semantic segmentation model paints as pavement.

Uses a Cityscapes-finetuned SegFormer model from Hugging Face Transformers,
downloaded automatically on first run.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

from yolo_filter import background_canvas, clear_stale_frames


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load_model(name):
    try:
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    except ImportError:
        raise SystemExit(
            "transformers is not installed. Run:\n"
            "  ./venv/bin/pip install -r requirements.txt"
        )
    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForSemanticSegmentation.from_pretrained(name)
    model.eval()
    return processor, model


def segment(processor, model, image, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, num_classes, h, w)
    upsampled = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    probs = upsampled.softmax(dim=1)[0]  # (num_classes, H, W)
    confidence, class_map = probs.max(dim=0)
    return class_map.cpu().numpy(), confidence.cpu().numpy()  # (H, W) each


def process_frame(processor, model, in_path, out_path, class_ids, background, device, min_confidence):
    image = Image.open(in_path).convert("RGB")
    np_image = np.array(image)

    class_map, confidence = segment(processor, model, image, device)
    mask = np.isin(class_map, class_ids) & (confidence > min_confidence)

    out_mode = "RGBA" if background == "transparent" else "RGB"
    canvas = background_canvas(image.size, out_mode, background)
    canvas_np = np.array(canvas)

    if out_mode == "RGB":
        canvas_np[mask] = np_image[mask]
    else:
        canvas_np[mask, :3] = np_image[mask]
        canvas_np[mask, 3] = 255

    Image.fromarray(canvas_np, out_mode).save(out_path)
    return int(mask.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="frames")
    ap.add_argument("--output-dir", default="frames_road")
    ap.add_argument("--pattern", default="frame_*.jpg", help="glob pattern for input frames")
    ap.add_argument("--model", default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    ap.add_argument("--classes", default="road", help="comma-separated class names to keep")
    ap.add_argument("--background", default="black", choices=["black", "white", "transparent"])
    ap.add_argument("--device", default="", help="'cpu', 'mps', 'cuda:0', ... (default: cpu)")
    ap.add_argument("--min-confidence", type=float, default=0.5, help="drop pixels below this softmax confidence (0-1)")
    ap.add_argument("--list-classes", action="store_true", help="print the model's class names and exit")
    args = ap.parse_args()

    processor, model = load_model(args.model)

    if args.list_classes:
        for idx, name in sorted(model.config.id2label.items()):
            print(f"{idx}\t{name}")
        return

    device = torch.device(args.device) if args.device else torch.device("cpu")
    model.to(device)

    name_to_id = {name.lower(): idx for idx, name in model.config.id2label.items()}
    wanted = {n.strip().lower() for n in args.classes.split(",") if n.strip()}
    unknown = wanted - set(name_to_id)
    if unknown:
        raise SystemExit(
            f"unknown class name(s): {', '.join(sorted(unknown))}\n"
            f"run with --list-classes to see what {args.model} supports"
        )
    class_ids = [name_to_id[n] for n in wanted]

    os.makedirs(args.output_dir, exist_ok=True)
    clear_stale_frames(args.output_dir)
    paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"no frames matching {args.pattern!r} in {args.input_dir}/")
    log(f"found {len(paths)} frames in {args.input_dir}/, keeping classes: {', '.join(sorted(wanted))}")

    ext = "png" if args.background == "transparent" else "jpg"
    for i, in_path in enumerate(paths):
        out_path = os.path.join(args.output_dir, f"frame_{i:05d}.{ext}")
        kept = process_frame(processor, model, in_path, out_path, class_ids, args.background, device, args.min_confidence)
        log(f"[{i}] {os.path.basename(in_path)}: kept {kept} px -> {out_path}")

    log(f"done. wrote {len(paths)} frames to {args.output_dir}/")
    log("assemble into video with e.g.:")
    log(f"  ffmpeg -framerate 24 -i {args.output_dir}/frame_%05d.{ext} -c:v libx264 -pix_fmt yuv420p out_road.mp4")


if __name__ == "__main__":
    main()
