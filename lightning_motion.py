#!/usr/bin/env python3
"""
lightning_motion.py -- reconstruct how a strike MOVED, not just its final path.

Where lightning_path.py collapses an event into one image, this tool preserves
and visualises the TIME dimension: which part of the channel lit first, which
direction it propagated, how brightness moved frame to frame.

A caution on physics: a single return stroke propagates in microseconds, far
faster than any frame captures. What you actually see moving between frames is
(a) different strokes re-illuminating parts of the channel, (b) the leader/
branches developing across the glow, and (c) the channel fading. So "motion"
here means the frame-to-frame evolution of the luminous channel, which is still
very informative about direction and development -- just not the microsecond
leader propagation itself.

Views it produces (pick any combination)
----------------------------------------
1. --timeline     : contact sheet of the strike frames in order, so you can
                    scan the evolution by eye.

2. --color-time   : ONE image where each frame's new channel pixels are tinted
                    by time (blue=early -> red=late). Shows direction of
                    development in a single picture WITHOUT losing time info.

3. --per-frame-new: writes each frame's NEWLY-lit pixels (what appeared since
                    the previous frame), so you see the growth increment by
                    increment.

4. --centroid     : tracks the brightness centroid of the channel per frame and
                    reports the motion vector (direction + speed in px/frame),
                    plus an arrow overlay. Tells you which way it moved.

5. --flow         : dense optical flow between consecutive strike frames,
                    saved as arrow overlays -- fine-grained motion field.

6. --animate      : writes an mp4 replaying just the strike frames slowed down,
                    the most direct way to SEE the movement.

Inputs
------
* a .npy stack (lightning_run.py --format npy) with an optional
  <base>_timestamps.json alongside (real per-frame times)
* a folder of png/jpg frames (frame order = filename order)

Examples
--------
  # see the development direction in one colour-coded image:
  python3 tests/lightning_motion.py --input event.npy --color-time evo.png

  # everything:
  python3 tests/lightning_motion.py --input event_frames \
      --timeline sheet.png --color-time evo.png --centroid track.png \
      --animate replay.mp4 --report
"""
import os
import sys
import glob
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lc_common import setup_logger                          # noqa: E402

import numpy as np                                          # noqa: E402
try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_event(path, log):
    """Return (frames[list of gray uint8], times[list of float or None])."""
    frames, times = [], None
    if os.path.isfile(path) and path.endswith(".npy"):
        arr = np.load(path)
        for i in range(arr.shape[0]):
            f = arr[i]
            if f.ndim == 3:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            frames.append(f.astype(np.uint8))
        ts_path = path[:-4] + "_timestamps.json"
        if os.path.exists(ts_path):
            with open(ts_path) as fh:
                times = json.load(fh)
            log.info("Loaded per-frame timestamps (%d).", len(times))
    elif os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.png")) +
                       glob.glob(os.path.join(path, "*.jpg")))
        for fp in files:
            g = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if g is not None:
                frames.append(g)
    else:
        raise FileNotFoundError(f"Cannot read frames from {path}")
    log.info("Loaded %d frames, size %s", len(frames),
             frames[0].shape if frames else "?")
    return frames, times


def strike_frames(frames, z=3.0):
    """Indices of frames containing the strike (99th-pct brightness spike)."""
    p99 = np.array([np.percentile(f, 99) for f in frames], dtype=np.float64)
    base = np.median(p99)
    mad = np.median(np.abs(p99 - base)) + 1e-6
    return [i for i, v in enumerate(p99) if (v - base) / mad > z]


def channel_of(frame, baseline, thresh_offset=25):
    """Binary channel mask for one frame vs a sky baseline."""
    resid = cv2.subtract(frame, baseline)
    _t, m = cv2.threshold(resid, thresh_offset, 255, cv2.THRESH_BINARY)
    return m


# --------------------------------------------------------------------------- #
# views
# --------------------------------------------------------------------------- #
def view_timeline(frames, idx, out, log, cols=6):
    """Contact sheet of the strike frames in order."""
    if not idx:
        log.warning("No strike frames to tile.")
        return
    h, w = frames[0].shape
    rows = (len(idx) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w), np.uint8)
    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        tile = frames[i].copy()
        cv2.putText(tile, f"#{i}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 255, 1)
        sheet[r*h:(r+1)*h, c*w:(c+1)*w] = tile
    cv2.imwrite(out, sheet)
    log.info("Wrote timeline contact sheet -> %s (%d frames)", out, len(idx))


def view_color_time(frames, idx, out, log):
    """
    One colour image: each strike frame's NEW channel pixels are tinted by
    time (blue=early -> red=late). Shows propagation direction in a single
    still without discarding the time ordering.
    """
    if not idx:
        log.warning("No strike frames for color-time.")
        return
    h, w = frames[0].shape
    baseline = np.median(np.stack(frames), axis=0).astype(np.uint8)
    canvas = np.zeros((h, w, 3), np.uint8)
    seen = np.zeros((h, w), bool)
    n = len(idx)
    for k, i in enumerate(idx):
        m = channel_of(frames[i], baseline) > 0
        new = m & ~seen
        seen |= m
        # colour by time: hue from 120 (blue-ish) down to 0 (red)
        frac = k / max(1, n - 1)
        hue = int((1 - frac) * 120)          # 120=green/blue early, 0=red late
        col = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        canvas[new] = col
    cv2.imwrite(out, canvas)
    log.info("Wrote colour-by-time evolution -> %s (blue=early, red=late)", out)


def view_per_frame_new(frames, idx, outdir, log):
    """Write each strike frame's newly-lit channel pixels as its own image."""
    if not idx:
        log.warning("No strike frames for per-frame-new.")
        return
    os.makedirs(outdir, exist_ok=True)
    baseline = np.median(np.stack(frames), axis=0).astype(np.uint8)
    seen = None
    for k, i in enumerate(idx):
        m = channel_of(frames[i], baseline)
        if seen is None:
            new = m
        else:
            new = cv2.subtract(m, seen)
        seen = m if seen is None else cv2.bitwise_or(seen, m)
        cv2.imwrite(os.path.join(outdir, f"new_{k:02d}_frame{i}.png"), new)
    log.info("Wrote per-frame new-growth images -> %s/", outdir)


def view_centroid(frames, idx, out, log, times=None):
    """
    Track the channel's brightness centroid per strike frame; report the net
    motion vector (direction + px/frame) and draw the track.
    """
    if len(idx) < 2:
        log.warning("Need >=2 strike frames to track centroid.")
        return
    baseline = np.median(np.stack(frames), axis=0).astype(np.uint8)
    pts = []
    for i in idx:
        resid = cv2.subtract(frames[i], baseline).astype(np.float64)
        s = resid.sum()
        if s <= 0:
            pts.append(None)
            continue
        ys, xs = np.mgrid[0:resid.shape[0], 0:resid.shape[1]]
        cx = (xs * resid).sum() / s
        cy = (ys * resid).sum() / s
        pts.append((cx, cy))

    valid = [(i, p) for i, p in zip(idx, pts) if p is not None]
    if len(valid) < 2:
        log.warning("Not enough valid centroids.")
        return

    # overlay track on the max-stack for context
    canvas = cv2.cvtColor(max_stack(frames), cv2.COLOR_GRAY2BGR)
    for a, b in zip(valid, valid[1:]):
        pa = tuple(map(int, a[1]))
        pb = tuple(map(int, b[1]))
        cv2.arrowedLine(canvas, pa, pb, (0, 0, 255), 2, tipLength=0.3)
    for _i, p in valid:
        cv2.circle(canvas, tuple(map(int, p)), 3, (0, 255, 0), -1)
    cv2.imwrite(out, canvas)

    (i0, p0), (i1, p1) = valid[0], valid[-1]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    dist = float(np.hypot(dx, dy))
    # image y grows downward; convert to compass-ish direction for readability
    ang = float(np.degrees(np.arctan2(-dy, dx)))   # 0=right,90=up
    nframes = i1 - i0
    if times and i1 < len(times) and i0 < len(times):
        dt = times[i1] - times[i0]
        speed = f"{dist/dt:.1f} px/s" if dt > 0 else "n/a"
    else:
        speed = f"{dist/max(1,nframes):.2f} px/frame"
    log.info("Centroid motion: net %.1f px, direction %.0f deg "
             "(0=right,90=up), speed %s over frames %d..%d",
             dist, ang, speed, i0, i1)
    log.info("Wrote centroid track overlay -> %s", out)


def view_flow(frames, idx, outdir, log, step=12):
    """Dense optical flow between consecutive strike frames, as arrow overlays."""
    if len(idx) < 2:
        log.warning("Need >=2 strike frames for flow.")
        return
    os.makedirs(outdir, exist_ok=True)
    for k in range(len(idx) - 1):
        a, b = frames[idx[k]], frames[idx[k+1]]
        flow = cv2.calcOpticalFlowFarneback(a, b, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        vis = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
        h, w = a.shape
        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow[y, x]
                if fx*fx + fy*fy > 1.0:
                    cv2.arrowedLine(vis, (x, y),
                                    (int(x+fx), int(y+fy)), (0, 0, 255), 1,
                                    tipLength=0.4)
        cv2.imwrite(os.path.join(outdir, f"flow_{k:02d}_{idx[k]}to{idx[k+1]}.png"),
                    vis)
    log.info("Wrote optical-flow overlays -> %s/", outdir)


def view_animate(frames, idx, out, log, fps=8, pad=2):
    """Replay the strike (with a little context padding) slowed down, as mp4."""
    if not idx:
        log.warning("No strike frames to animate.")
        return
    lo = max(0, min(idx) - pad)
    hi = min(len(frames), max(idx) + pad + 1)
    seq = frames[lo:hi]
    h, w = seq[0].shape
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h),
                         isColor=False)
    for j, f in enumerate(seq):
        fr = f.copy()
        tag = "STRIKE" if (lo + j) in idx else ""
        cv2.putText(fr, f"{lo+j} {tag}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
        vw.write(fr)
    vw.release()
    log.info("Wrote slowed replay -> %s (%d frames @ %d fps)", out, len(seq), fps)


def max_stack(frames):
    out = frames[0].copy()
    for f in frames[1:]:
        np.maximum(out, f, out=out)
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Reconstruct lightning MOTION / direction from event frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", required=True,
                    help=".npy stack (+ _timestamps.json) or png/jpg folder")
    ap.add_argument("--z", type=float, default=3.0,
                    help="strike-detection sensitivity (lower = more frames)")
    ap.add_argument("--all-frames", action="store_true",
                    help="treat every frame as part of the event (skip strike "
                         "detection)")

    ap.add_argument("--timeline", metavar="OUT.png",
                    help="contact sheet of strike frames in order")
    ap.add_argument("--color-time", metavar="OUT.png",
                    help="single image, channel tinted by time (blue->red)")
    ap.add_argument("--per-frame-new", metavar="OUTDIR",
                    help="write each frame's newly-lit pixels")
    ap.add_argument("--centroid", metavar="OUT.png",
                    help="track brightness centroid + report motion vector")
    ap.add_argument("--flow", metavar="OUTDIR",
                    help="optical-flow arrow overlays between strike frames")
    ap.add_argument("--animate", metavar="OUT.mp4",
                    help="slowed replay of the strike frames")
    ap.add_argument("--anim-fps", type=int, default=8)
    ap.add_argument("--report", action="store_true",
                    help="log which frames contain the strike")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    log = setup_logger("lightning_motion", args.log)
    if not _HAVE_CV2:
        log.error("OpenCV not available.")
        return 2

    frames, times = load_event(args.input, log)
    if not frames:
        log.error("No frames.")
        return 3

    idx = list(range(len(frames))) if args.all_frames else strike_frames(frames, args.z)
    if args.report or not idx:
        log.info("Strike frames: %s (of %d)", idx if idx else "NONE", len(frames))
    if not idx:
        log.warning("No strike detected; re-run with --all-frames or lower --z.")
        # still allow views on all frames if user forces later
        return 0

    if args.timeline:
        view_timeline(frames, idx, args.timeline, log)
    if args.color_time:
        view_color_time(frames, idx, args.color_time, log)
    if args.per_frame_new:
        view_per_frame_new(frames, idx, args.per_frame_new, log)
    if args.centroid:
        view_centroid(frames, idx, args.centroid, log, times=times)
    if args.flow:
        view_flow(frames, idx, args.flow, log)
    if args.animate:
        view_animate(frames, idx, args.animate, log, fps=args.anim_fps)

    if not any([args.timeline, args.color_time, args.per_frame_new,
                args.centroid, args.flow, args.animate]):
        log.info("No view selected. Add e.g. --color-time evo.png or --animate "
                 "replay.mp4. Use --help to see all views.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
