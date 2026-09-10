#!/usr/bin/env python3
"""
Layer road_filter.py and yolo_filter.py on the same black canvas: the
road/sidewalk (cement) cutout goes down first, then the objects YOLO
detects are painted on top of it.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

from road_filter import load_model as load_road_model
from road_filter import segment
from yolo_filter import background_canvas, clear_stale_frames, label_font
from yolo_filter import load_model as load_yolo_model


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def process_frame(road_processor, road_model, road_class_ids, road_device, road_min_confidence,
                   yolo_model, yolo_args, in_path, out_path, background, labels, font):
    image = Image.open(in_path).convert("RGB")
    np_image = np.array(image)

    out_mode = "RGBA" if background == "transparent" else "RGB"
    canvas_np = np.array(background_canvas(image.size, out_mode, background))

    # layer 1: road / sidewalk
    class_map, confidence = segment(road_processor, road_model, image, road_device)
    road_mask = np.isin(class_map, road_class_ids) & (confidence > road_min_confidence)
    if out_mode == "RGB":
        canvas_np[road_mask] = np_image[road_mask]
    else:
        canvas_np[road_mask, :3] = np_image[road_mask]
        canvas_np[road_mask, 3] = 255

    # layer 2: yolo objects, painted on top
    results = yolo_model.predict(
        np_image,
        conf=yolo_args["conf"],
        classes=yolo_args["class_ids"],
        device=yolo_args["device"] or None,
        retina_masks=True,
        verbose=False,
    )[0]

    boxes = results.boxes
    masks = results.masks
    kept = 0
    if boxes is not None:
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].round().int().tolist()
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

    if labels and kept:
        draw = ImageDraw.Draw(out_image)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].round().int().tolist()
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            name = yolo_model.names[cls_id]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            draw.text((x1 + 2, max(0, y1 - 16)), f"{name} {conf:.2f}", fill=(255, 0, 0), font=font)

    out_image.save(out_path)
    return int(road_mask.sum()), kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="frames")
    ap.add_argument("--output-dir", default="frames_models_added_up")
    ap.add_argument("--pattern", default="frame_*.jpg", help="glob pattern for input frames")
    ap.add_argument("--road-model", default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    ap.add_argument("--road-classes", default="road,sidewalk")
    ap.add_argument("--road-device", default="", help="'cpu', 'mps', 'cuda:0', ... (default: cpu)")
    ap.add_argument("--road-min-confidence", type=float, default=0.5, help="drop road/sidewalk pixels below this softmax confidence (0-1)")
    ap.add_argument("--yolo-model", default="yolov8n-seg.pt")
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--yolo-classes", default="", help="comma-separated class names to keep (default: all)")
    ap.add_argument("--yolo-device", default="", help="'cpu', 'mps', 'cuda:0', ... (default: let ultralytics choose)")
    ap.add_argument("--background", default="black", choices=["black", "white", "transparent"])
    ap.add_argument("--labels", action="store_true", help="draw box outline + class name/confidence over the yolo layer")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    clear_stale_frames(args.output_dir)

    log(f"loading road segmentation model {args.road_model}...")
    road_processor, road_model = load_road_model(args.road_model)
    road_device = torch.device(args.road_device) if args.road_device else torch.device("cpu")
    road_model.to(road_device)
    road_name_to_id = {name.lower(): idx for idx, name in road_model.config.id2label.items()}
    road_wanted = {n.strip().lower() for n in args.road_classes.split(",") if n.strip()}
    unknown = road_wanted - set(road_name_to_id)
    if unknown:
        raise SystemExit(f"unknown road class name(s): {', '.join(sorted(unknown))}")
    road_class_ids = [road_name_to_id[n] for n in road_wanted]

    log(f"loading yolo model {args.yolo_model}...")
    yolo_model = load_yolo_model(args.yolo_model)
    yolo_class_ids = None
    if args.yolo_classes:
        yolo_wanted = {n.strip().lower() for n in args.yolo_classes.split(",") if n.strip()}
        yolo_name_to_id = {name.lower(): idx for idx, name in yolo_model.names.items()}
        unknown_y = yolo_wanted - set(yolo_name_to_id)
        if unknown_y:
            raise SystemExit(f"unknown yolo class name(s): {', '.join(sorted(unknown_y))}")
        yolo_class_ids = [yolo_name_to_id[n] for n in yolo_wanted]
    yolo_args = {"conf": args.yolo_conf, "class_ids": yolo_class_ids, "device": args.yolo_device}

    font = label_font()

    paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"no frames matching {args.pattern!r} in {args.input_dir}/")
    log(f"found {len(paths)} frames in {args.input_dir}/, road classes: {', '.join(sorted(road_wanted))}")

    ext = "png" if args.background == "transparent" else "jpg"
    for i, in_path in enumerate(paths):
        out_path = os.path.join(args.output_dir, f"frame_{i:05d}.{ext}")
        road_px, kept = process_frame(
            road_processor, road_model, road_class_ids, road_device, args.road_min_confidence,
            yolo_model, yolo_args, in_path, out_path, args.background, args.labels, font,
        )
        log(f"[{i}] {os.path.basename(in_path)}: road {road_px}px + {kept} object(s) -> {out_path}")

    log(f"done. wrote {len(paths)} frames to {args.output_dir}/")
    log("assemble into video with e.g.:")
    log(f"  ffmpeg -framerate 24 -i {args.output_dir}/frame_%05d.{ext} -c:v libx264 -pix_fmt yuv420p out_combined.mp4")


if __name__ == "__main__":
    main()
