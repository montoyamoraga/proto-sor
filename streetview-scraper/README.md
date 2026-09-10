# streetview-scraper

Downloads a sequence of Google Street View frames along a road route between
two points, for stitching into a flythrough video.

No Google API key needed. It uses only free, keyless services:

- [OSRM](http://router.project-osrm.org) for road routing (so frames follow
  actual streets, not a straight line)
- Google's own undocumented panorama-search and raw-tile endpoints (the same
  ones the Street View web viewer loads in your browser) to find and fetch
  imagery

**Heads up:** these are not the official, billed Street View Static API.
They're unsupported, can change or start blocking requests at any time, and
scraping them is outside what Google's terms of service cover for the
official API. Keep `--delay` reasonable, don't parallelize requests, and use
this for personal projects rather than anything at scale or in production.

## Setup

Your system Python may have an SSL library too old to talk to some of these
hosts (seen with macOS's Xcode-bundled Python 3.9 / LibreSSL). Use a modern
Python (Homebrew's `python3`, pyenv, etc.) for the venv:

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Usage

```bash
./venv/bin/python3 scraper.py \
  --start "37.7749,-122.4194" \
  --end "37.7768,-122.4172" \
  --spacing 10 \
  --output-dir frames
```

`--start`/`--end` accept either `lat,lon` or a free-text address (geocoded
for free via OpenStreetMap Nominatim).

Then assemble the frames into a video with ffmpeg:

```bash
ffmpeg -framerate 24 -i frames/frame_%05d.jpg -c:v libx264 -pix_fmt yuv420p out.mp4
```

### Useful flags

| flag | default | meaning |
|---|---|---|
| `--spacing` | 10 | meters between sampled frames along the route |
| `--profile` | driving | OSRM routing profile: driving / walking / cycling |
| `--zoom` | 3 | Street View tile zoom, 0 (blurry/fast) .. 5 (full-res/slow) |
| `--width` / `--height` | 1280x720 | output frame size |
| `--fov` | 90 | horizontal field of view in degrees |
| `--pitch` | 0 | camera tilt, +up / -down |
| `--heading-offset` | 0 | degrees added to the computed travel heading, if frames look rotated |
| `--delay` | 0.3 | seconds between requests (be polite) |
| `--max-frames` | 0 (no limit) | stop early, useful for testing |

Frames are only emitted at distinct panoramas, so if `--spacing` is smaller
than the real gap between Street View panoramas on a given street you'll get
fewer frames than sample points — that's expected, not a bug. Start with a
small `--max-frames` to sanity-check heading/framing before running a full
route.

## Filtering frames through YOLO

`yolo_filter.py` runs each frame through [Ultralytics
YOLO](https://docs.ultralytics.com) and keeps only the pixels of the objects
it detects (cars, people, bicycles, ...), replacing everything else with a
flat background — the "only what the model sees" look used in pieces like
Andreas Refsgaard's computer-vision work.

```bash
./venv/bin/python3 yolo_filter.py \
  --input-dir frames \
  --output-dir frames_yolo
```

It defaults to `yolov8n-seg.pt`, a segmentation model, so each object is cut
out along its silhouette rather than as a rectangle. The first run downloads
the model weights (a few MB) into this directory automatically.

| flag | default | meaning |
|---|---|---|
| `--input-dir` | frames | folder of source frames |
| `--output-dir` | frames_yolo | where cutout frames are written |
| `--pattern` | frame_*.jpg | glob for input frames |
| `--model` | yolov8n-seg.pt | any Ultralytics checkpoint; use a plain (non `-seg`) model for bounding-box crops instead of pixel-accurate cutouts |
| `--conf` | 0.25 | detection confidence threshold |
| `--classes` | (all) | comma-separated class names to keep, e.g. `car,person,bicycle` |
| `--background` | black | `black`, `white`, or `transparent` (writes PNGs) |
| `--labels` | off | draw the box outline + class name/confidence over each cutout |
| `--device` | auto | `cpu`, `mps` (Apple Silicon), `cuda:0`, ... |

Then assemble into a video the same way:

```bash
ffmpeg -framerate 24 -i frames_yolo/frame_%05d.jpg -c:v libx264 -pix_fmt yuv420p out_yolo.mp4
```

## Filtering frames down to just the road

`road_filter.py` keeps the opposite kind of thing: not discrete objects, but
the ground plane itself. YOLO has no notion of "road" (it's not an object),
so this uses a semantic segmentation model instead -- one trained on
[Cityscapes](https://www.cityscapes-dataset.com/), which labels every pixel
of a street scene, including `road` (asphalt) and `sidewalk` (cement).
Everything else -- buildings, sky, vehicles, people -- is discarded.

```bash
./venv/bin/python3 road_filter.py \
  --input-dir frames \
  --output-dir frames_road
```

It defaults to `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`, a small
SegFormer checkpoint downloaded from Hugging Face on first run. For cleaner
mask edges at the cost of speed, try a bigger checkpoint, e.g. `--model
nvidia/segformer-b5-finetuned-cityscapes-1024-1024`.

| flag | default | meaning |
|---|---|---|
| `--input-dir` | frames | folder of source frames |
| `--output-dir` | frames_road | where filtered frames are written |
| `--pattern` | frame_*.jpg | glob for input frames |
| `--model` | segformer-b0-...-cityscapes | any Hugging Face semantic segmentation checkpoint with matching label names |
| `--classes` | road,sidewalk | comma-separated class names to keep |
| `--background` | black | `black`, `white`, or `transparent` (writes PNGs) |
| `--device` | cpu | `cpu`, `mps` (Apple Silicon), `cuda:0`, ... |
| `--list-classes` | off | print the model's class names and exit |

Then assemble into a video the same way:

```bash
ffmpeg -framerate 24 -i frames_road/frame_%05d.jpg -c:v libx264 -pix_fmt yuv420p out_road.mp4
```

## Layering both filters

`combined_filter.py` runs both models on each frame and stacks their
cutouts on the same canvas: the road/sidewalk layer from `road_filter.py`
goes down first, then the objects `yolo_filter.py` finds are painted on
top of it.

```bash
./venv/bin/python3 combined_filter.py \
  --input-dir frames \
  --output-dir frames_models_added_up
```

It accepts the same tuning flags as the two filters it combines, prefixed
by which model they apply to: `--road-model`, `--road-classes`,
`--road-device` for the segmentation layer; `--yolo-model`, `--yolo-conf`,
`--yolo-classes`, `--yolo-device` for the object layer; plus the shared
`--background` and `--labels` flags from `yolo_filter.py`.

Then assemble into a video the same way:

```bash
ffmpeg -framerate 24 -i frames_models_added_up/frame_%05d.jpg -c:v libx264 -pix_fmt yuv420p out_combined.mp4
```

## Narrating what YOLO sees

`yolo_labels.py` doesn't touch the pixels at all -- it runs YOLO over each
frame and writes out the list of object names it detects, one line per
frame, ordered left-to-right the way they appear in the shot. Meant to be
fed to a text-to-speech engine that reads off everything the model sees as
the frames go by.

```bash
./venv/bin/python3 yolo_labels.py \
  --input-dir frames \
  --output frames_yolo_labels.txt
```

```
frame_00006: car, car, car, car, car, person, person, car, car, person, ...
frame_00007: traffic light, person, car, car, car
frame_00008: person, person, stop sign
```

| flag | default | meaning |
|---|---|---|
| `--input-dir` | frames | folder of source frames |
| `--output` | frames_yolo_labels.txt | where the transcript is written |
| `--pattern` | frame_*.jpg | glob for input frames |
| `--model` | yolov8n.pt | any Ultralytics detection model (masks aren't needed here) |
| `--conf` | 0.25 | detection confidence threshold |
| `--classes` | (all) | comma-separated class names to keep |
| `--order` | left-to-right | order objects appear within a frame's line: scanning left-to-right, or `confidence` (most confident first) |
| `--format` | labeled | `labeled` prefixes each line with the frame name; `plain` writes just the object list, one frame per line, ready to pipe straight into a TTS engine |
| `--empty-placeholder` | (nothing) | text written for a frame with no detections |

## Stitching a narrated video

`narrate_video.py` combines the last two steps into a finished video, at a
fixed, fast framerate: it runs YOLO on each frame and, for every frame,
synthesizes every distinct object it found and mixes those words together
so they play back **all at once, overlapping** -- not read out one after
another. A frame with ten cars says "car" once (pitched up, see below),
not ten times, so a busy frame doesn't cost ten times as much narration
time as a quiet one, and the video stays short regardless of how much is
in frame.

A class that shows up more than once within a frame is pitched (and sped)
up in proportion to how many there are, so "one car" and "five cars" are
audibly different without anyone having to say the number out loud.

```bash
./venv/bin/python3 narrate_video.py \
  --frames-dir frames_models_added_up \
  --fps 5 \
  --output out_narrated.mp4
```

By default it detects objects on the same frames it displays. Pass
`--labels-dir` to detect on a different (e.g. un-filtered) folder instead
-- it must have the exact same number of frames, in the same order, as
`--frames-dir`.

| flag | default | meaning |
|---|---|---|
| `--frames-dir` | frames_models_added_up | frames to show on screen |
| `--labels-dir` | (same as --frames-dir) | frames to run YOLO on for narration |
| `--frame-pattern` | frame_%05d.jpg | printf-style pattern ffmpeg reads the frames with |
| `--pattern` | frame_*.jpg | glob pattern used to count/list the frames |
| `--output` | out_narrated.mp4 | output video path |
| `--fps` | 5 | playback framerate |
| `--model` | yolov8n.pt | any Ultralytics detection model |
| `--conf` | 0.25 | detection confidence threshold |
| `--classes` | (all) | comma-separated class names to keep |
| `--lang` | en | `en` or `es` -- translates COCO class names (via `coco_es.py`) and picks a matching default voice |
| `--voice` | Samantha | any voice from `say -v ?` (default is Mónica for `--lang es`) |
| `--rate` | 400 | words per minute (say's own default is ~175-200) |
| `--pitch-step` | 0.05 | pitch/speed multiplier added per extra duplicate of a class within one frame (uncapped) |

For a Spanish narration: `--lang es` (defaults to the `Mónica` voice; override with `--voice`).
`yolo_labels.py` also accepts `--lang es` for a Spanish transcript.

Requires macOS (`say`) and `ffmpeg` on `PATH` (`brew install ffmpeg`).
