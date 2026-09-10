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
