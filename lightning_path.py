#!/usr/bin/env python3
"""
lightning_path.py -- reconstruct the lightning channel (path) from event frames.

A single return stroke propagates in microseconds -- no frame rate freezes the
leader itself. What you capture is the luminous channel glowing for a few to
tens of milliseconds, often flickering across several frames as multiple
strokes re-illuminate the same path. This tool turns those frames into a
picture of the PATH.

Techniques provided
-------------------
1. max-stack     : per-pixel maximum across all frames. The channel is the
                   brightest feature in every frame it appears in, so the
                   composite shows the full path against the dark sky, even if
                   the channel flickered across frames. This is the workhorse.

2. diff-stack    : subtract a "sky baseline" (median of frames) from each frame
                   before max-stacking. Removes static bright objects (lights,
                   horizon) so only the transient channel remains.

3. channel-mask  : threshold + thin the stacked image to a skeleton, giving a
                   1-pixel-wide trace of the path you can overlay or measure.

4. which-frames  : report which frames actually contain the strike (by frame
                   brightness spike), so you know the stroke timing.

Inputs it understands
---------------------
* a .npy stack saved by lightning_run.py --format npy
* a folder of PNG frames (--format png)
* a directory of raw .jpg frames (from the raw MJPEG path)

Examples
--------
  python3 tests/lightning_path.py --input event_2026..._frames --out path.png
  python3 tests/lightning_path.py --input event_2026....npy --method diff \
      --out path.png --mask mask.png --report
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


def load_frames(path, log):
    """Return a list of grayscale uint8 frames from npy / png folder / jpg folder."""
    frames = []
    if os.path.isfile(path) and path.endswith(".npy"):
        arr = np.load(path)
        log.info("Loaded npy stack %s shape=%s", path, arr.shape)
        for i in range(arr.shape[0]):
            f = arr[i]
            if f.ndim == 3:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            frames.append(f.astype(np.uint8))
        return frames

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.png")) +
                       glob.glob(os.path.join(path, "*.jpg")))
        log.info("Loading %d image frames from %s", len(files), path)
        for fp in files:
            f = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if f is not None:
                frames.append(f)
        return frames

    raise FileNotFoundError(f"Cannot read frames from {path}")


def which_frames_have_strike(frames, log, z=3.0):
    """
    Identify frames containing the strike by a brightness spike: a frame whose
    mean (or 99th-percentile) brightness is well above the baseline.
    Returns list of indices.
    """
    p99 = np.array([np.percentile(f, 99) for f in frames], dtype=np.float64)
    base = np.median(p99)
    mad = np.median(np.abs(p99 - base)) + 1e-6
    hot = [i for i, v in enumerate(p99) if (v - base) / mad > z]
    log.info("Strike frames (brightness spike): %s of %d total",
             hot if hot else "NONE DETECTED", len(frames))
    return hot


def max_stack(frames):
    """Per-pixel maximum across all frames."""
    out = frames[0].copy()
    for f in frames[1:]:
        np.maximum(out, f, out=out)
    return out


def diff_stack(frames):
    """
    Subtract a median 'sky baseline' from each frame, then max-stack the
    positive residuals. Removes static bright objects; keeps the transient
    channel.
    """
    stack = np.stack(frames).astype(np.int16)
    baseline = np.median(stack, axis=0).astype(np.int16)
    resid = np.clip(stack - baseline, 0, 255).astype(np.uint8)
    out = resid[0].copy()
    for i in range(1, resid.shape[0]):
        np.maximum(out, resid[i], out=out)
    return out


def channel_mask(stacked, thresh=None):
    """
    Threshold the stacked image and thin to a skeleton-ish trace of the path.
    Returns a binary uint8 mask.
    """
    if thresh is None:
        # Otsu picks a data-driven threshold between sky and channel.
        _t, binmask = cv2.threshold(stacked, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _t, binmask = cv2.threshold(stacked, thresh, 255, cv2.THRESH_BINARY)
    # close small gaps along the channel, then thin
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binmask = cv2.morphologyEx(binmask, cv2.MORPH_CLOSE, kernel)
    try:
        thin = cv2.ximgproc.thinning(binmask)   # needs opencv-contrib
        return thin
    except Exception:
        return binmask


def main():
    ap = argparse.ArgumentParser(description="Reconstruct lightning path from frames")
    ap.add_argument("--input", required=True,
                    help=".npy stack, or folder of png/jpg frames")
    ap.add_argument("--method", choices=["max", "diff"], default="max",
                    help="max-stack (simple) or diff-stack (removes static lights)")
    ap.add_argument("--out", default="path.png", help="output composite image")
    ap.add_argument("--mask", default=None, help="also write a channel skeleton mask")
    ap.add_argument("--thresh", type=int, default=None,
                    help="fixed threshold for the mask (default: Otsu)")
    ap.add_argument("--report", action="store_true",
                    help="report which frames contain the strike")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    log = setup_logger("lightning_path", args.log)
    if not _HAVE_CV2:
        log.error("OpenCV not available.")
        return 2

    frames = load_frames(args.input, log)
    if not frames:
        log.error("No frames loaded.")
        return 3
    log.info("Loaded %d frames, size %s", len(frames), frames[0].shape)

    if args.report:
        which_frames_have_strike(frames, log)

    if args.method == "diff":
        stacked = diff_stack(frames)
    else:
        stacked = max_stack(frames)

    cv2.imwrite(args.out, stacked)
    log.info("Wrote path composite -> %s", args.out)

    if args.mask:
        m = channel_mask(stacked, args.thresh)
        cv2.imwrite(args.mask, m)
        log.info("Wrote channel mask -> %s", args.mask)
        n_channel = int((m > 0).sum())
        log.info("Channel pixels in mask: %d", n_channel)

    return 0


if __name__ == "__main__":
    sys.exit(main())
