#!/usr/bin/env python3
"""
Download a sequence of Google Street View frames following a road route
between two points, for stitching into a flythrough video.

Uses free, keyless services only:
  - OSRM (router.project-osrm.org) for road routing
  - Google's undocumented panorama-search + raw tile endpoints (the same
    ones the Street View web viewer itself uses) for imagery

These tile/search endpoints are not the official, billed Street View
Static API, are not documented or supported by Google, and can change or
start blocking requests at any time. Be polite: keep --delay reasonable
and don't hammer the endpoints in parallel.
"""
import argparse
import io
import json
import math
import os
import re
import sys
import time

import numpy as np
import requests
from PIL import Image

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

OSRM_URL = "https://router.project-osrm.org/route/v1/{profile}/{coords}"
PANO_SEARCH_URL = "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch"
TILE_URL = "https://cbk0.google.com/cbk"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

ZOOM_DIMS = {0: (256, 512), 1: (512, 1024), 2: (1024, 2048), 3: (2048, 4096), 4: (4096, 8192), 5: (8192, 16384)}
TILE_SIZE = 512

EARTH_RADIUS_M = 6371000.0


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# geo helpers
# ---------------------------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlambda)
    return math.degrees(math.atan2(y, x)) % 360


def parse_point(text, session):
    text = text.strip()
    m = re.match(r"^\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*$", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return geocode(text, session)


def geocode(address, session):
    log(f"geocoding '{address}' via Nominatim...")
    resp = session.get(
        NOMINATIM_URL,
        params={"q": address, "format": "json", "limit": 1},
        headers={"User-Agent": "streetview-scraper (personal use)"},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise SystemExit(f"could not geocode address: {address!r}")
    time.sleep(1)  # respect Nominatim's 1 req/sec usage policy
    return float(results[0]["lat"]), float(results[0]["lon"])


# ---------------------------------------------------------------------------
# routing (OSRM, free/keyless)
# ---------------------------------------------------------------------------

def fetch_route(start, end, profile, session):
    coords = f"{start[1]},{start[0]};{end[1]},{end[0]}"
    url = OSRM_URL.format(profile=profile, coords=coords)
    resp = session.get(url, params={"overview": "full", "geometries": "geojson"}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise SystemExit(f"OSRM could not find a route: {data.get('code')} {data.get('message', '')}")
    coords = data["routes"][0]["geometry"]["coordinates"]  # [[lon, lat], ...]
    return [(lat, lon) for lon, lat in coords]


def sample_route(route_latlon, spacing_m):
    """Walk the polyline and emit points every spacing_m meters."""
    if len(route_latlon) < 2:
        return list(route_latlon)

    points = [route_latlon[0]]
    carry = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(route_latlon, route_latlon[1:]):
        seg_len = haversine_m(lat1, lon1, lat2, lon2)
        if seg_len == 0:
            continue
        dist_into_seg = spacing_m - carry
        while dist_into_seg <= seg_len:
            f = dist_into_seg / seg_len
            points.append((lat1 + (lat2 - lat1) * f, lon1 + (lon2 - lon1) * f))
            dist_into_seg += spacing_m
        carry = seg_len - (dist_into_seg - spacing_m)
    if points[-1] != route_latlon[-1]:
        points.append(route_latlon[-1])
    return points


# ---------------------------------------------------------------------------
# Street View panorama lookup (undocumented, keyless)
# ---------------------------------------------------------------------------

def find_panorama(lat, lon, session, radius_m=50):
    pb = (
        "!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d%.8f!4d%.8f!2d%d!"
        "3m10!2m2!1sen!2sUS!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2"
    ) % (lat, lon, radius_m)
    resp = session.get(
        PANO_SEARCH_URL,
        params={"pb": pb, "callback": "_xdc_._scraper"},
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    resp.raise_for_status()
    text = resp.text
    try:
        data = json.loads(text[text.index("(") + 1:text.rindex(")")])
    except (ValueError, json.JSONDecodeError):
        return None

    block = data[1] if len(data) > 1 else None
    if not block or not block[1]:
        return None

    pano_id = block[1][1]
    try:
        loc = block[5][0][1][0]
        pano_lat, pano_lon = loc[2], loc[3]
    except (IndexError, TypeError):
        pano_lat, pano_lon = lat, lon

    return {"pano_id": pano_id, "lat": pano_lat, "lon": pano_lon}


# ---------------------------------------------------------------------------
# tile download + equirectangular stitching
# ---------------------------------------------------------------------------

def fetch_tile(pano_id, zoom, x, y, session):
    resp = session.get(
        TILE_URL,
        params={
            "output": "tile",
            "panoid": pano_id,
            "zoom": zoom,
            "x": x,
            "y": y,
            "cb_client": "maps_sv.tactile",
            "nbt": 1,
            "fov": 180,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def fetch_panorama(pano_id, zoom, session, tile_delay=0.0):
    height, width = ZOOM_DIMS[zoom]
    tiles_x, tiles_y = width // TILE_SIZE, height // TILE_SIZE
    canvas = Image.new("RGB", (width, height))
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile = fetch_tile(pano_id, zoom, tx, ty, session)
            canvas.paste(tile, (tx * TILE_SIZE, ty * TILE_SIZE))
            if tile_delay:
                time.sleep(tile_delay)
    return canvas


# ---------------------------------------------------------------------------
# equirectangular -> rectilinear (perspective) projection
# ---------------------------------------------------------------------------

def equirect_to_perspective(equirect_np, out_w, out_h, yaw_deg, pitch_deg, hfov_deg):
    """yaw_deg: compass heading (0=N, clockwise) to look towards. pitch_deg: up/down tilt."""
    H, W = equirect_np.shape[:2]
    aspect = out_h / out_w
    hfov = math.radians(hfov_deg)
    vfov = 2 * math.atan(math.tan(hfov / 2) * aspect)

    xi, yj = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
    x = xi * math.tan(hfov / 2)
    y = yj * math.tan(vfov / 2)
    z = np.ones_like(x)

    dx, dy, dz = x, -y, z
    norm = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    dx, dy, dz = dx / norm, dy / norm, dz / norm

    pitch0 = math.radians(pitch_deg)
    dy2 = dy * math.cos(pitch0) - dz * math.sin(pitch0)
    dz2 = dy * math.sin(pitch0) + dz * math.cos(pitch0)

    yaw0 = math.radians(yaw_deg)
    dx3 = dx * math.cos(yaw0) + dz2 * math.sin(yaw0)
    dz3 = -dx * math.sin(yaw0) + dz2 * math.cos(yaw0)
    dy3 = dy2

    yaw = np.degrees(np.arctan2(dx3, dz3)) % 360
    pitch = np.degrees(np.arcsin(np.clip(dy3, -1, 1)))

    u = np.clip((yaw / 360.0) * W, 0, W - 1).astype(np.int32)
    v = np.clip(((90.0 - pitch) / 180.0) * H, 0, H - 1).astype(np.int32)
    return equirect_np[v, u]


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="'lat,lon' or an address")
    ap.add_argument("--end", required=True, help="'lat,lon' or an address")
    ap.add_argument("--output-dir", default="frames")
    ap.add_argument("--spacing", type=float, default=10.0, help="meters between sampled frames along the route")
    ap.add_argument("--profile", default="driving", choices=["driving", "walking", "cycling"])
    ap.add_argument("--zoom", type=int, default=3, choices=sorted(ZOOM_DIMS), help="pano tile zoom (0=lowres..5=fullres)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=90.0, help="horizontal field of view in degrees")
    ap.add_argument("--pitch", type=float, default=0.0, help="camera tilt in degrees, +up/-down")
    ap.add_argument("--heading-offset", type=float, default=0.0, help="constant degrees added to computed travel heading, for calibration")
    ap.add_argument("--delay", type=float, default=0.3, help="seconds to sleep between pano-search / tile requests")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    session = requests.Session()

    start = parse_point(args.start, session)
    end = parse_point(args.end, session)
    log(f"start={start} end={end}")

    log(f"fetching {args.profile} route from OSRM...")
    route = fetch_route(start, end, args.profile, session)
    total_m = sum(haversine_m(*a, *b) for a, b in zip(route, route[1:]))
    log(f"route has {len(route)} vertices, ~{total_m:.0f}m long")

    sample_points = sample_route(route, args.spacing)
    log(f"sampled {len(sample_points)} points every {args.spacing}m")

    frame_idx = 0
    last_pano_id = None
    for i, (lat, lon) in enumerate(sample_points):
        if args.max_frames and frame_idx >= args.max_frames:
            log("hit --max-frames, stopping")
            break

        pano = find_panorama(lat, lon, session)
        time.sleep(args.delay)
        if pano is None:
            log(f"[{i}] no imagery near ({lat:.6f},{lon:.6f}), skipping")
            continue
        if pano["pano_id"] == last_pano_id:
            continue
        last_pano_id = pano["pano_id"]

        # heading = bearing toward the next distinct sample point (direction of travel)
        look_at = sample_points[min(i + 1, len(sample_points) - 1)]
        heading = bearing_deg(pano["lat"], pano["lon"], look_at[0], look_at[1]) + args.heading_offset

        try:
            equirect = fetch_panorama(pano["pano_id"], args.zoom, session, tile_delay=args.delay)
        except requests.HTTPError as e:
            log(f"[{i}] failed to fetch pano {pano['pano_id']}: {e}")
            continue

        frame = equirect_to_perspective(
            np.array(equirect), args.width, args.height, heading, args.pitch, args.fov
        )
        out_path = os.path.join(args.output_dir, f"frame_{frame_idx:05d}.jpg")
        Image.fromarray(frame).save(out_path, quality=92)
        log(f"[{i}] frame {frame_idx:05d} <- pano {pano['pano_id']} heading={heading:.1f} -> {out_path}")
        frame_idx += 1

    log(f"done. wrote {frame_idx} frames to {args.output_dir}/")
    log(f"assemble into video with e.g.:")
    log(f"  ffmpeg -framerate 24 -i {args.output_dir}/frame_%05d.jpg -c:v libx264 -pix_fmt yuv420p out.mp4")


if __name__ == "__main__":
    main()
