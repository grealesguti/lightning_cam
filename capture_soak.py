#!/usr/bin/env python3
"""
capture_soak.py -- thoroughly test the RAW capture path for lightning duty.

test_camera / camera_benchmark measure a mode for a few seconds. This runs the
RAW MJPEG path the way the recorder will, for a long stretch, and reports the
things that actually decide whether you'll catch a strike cleanly:

  * sustained FPS over minutes (not a 3 s burst)
  * frame-interval consistency: mean, std-dev, worst gap. A long gap = a moment
    where a strike could fall between frames. This is the metric that matters
    most for motion capture.
  * dropped/failed reads over time
  * JPEG size stability (a proxy for exposure/scene stability; wild swings mean
    auto-exposure is still active and will corrupt motion analysis)
  * periodic decode-check: every N seconds it decodes one frame and reports
    sharpness + brightness, so you see if focus/exposure drift during a long run
  * memory budget: projects RAM needed for your pre+post ring buffer at the
    measured frame size and rate

Run this for several minutes before trusting an overnight capture.

Examples
--------
  python3 tests/capture_soak.py --seconds 300
  python3 tests/capture_soak.py --seconds 600 --width 640 --height 400 --fps 120 \
      --exposure 200 --gain 20 --pre 1.5 --post 1.5
"""
import os
import sys
import time
import argparse
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lc_common import setup_logger, human_bytes             # noqa: E402
from lc_camera import RawMJPEGCamera, CameraV4L2            # noqa: E402

import numpy as np                                          # noqa: E402
try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False


def main():
    ap = argparse.ArgumentParser(description="Long soak test of the raw capture path")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--fps", type=int, default=120)
    ap.add_argument("--exposure", type=int, default=None,
                    help="manual exposure (v4l2 units); omit to leave as-is")
    ap.add_argument("--gain", type=int, default=None, help="manual gain")
    ap.add_argument("--auto-exposure", action="store_true",
                    help="allow auto-exposure (NOT recommended for motion)")
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--check-every", type=float, default=30,
                    help="seconds between decode quality checks")
    ap.add_argument("--pre", type=float, default=1.5, help="ring pre-roll (for RAM calc)")
    ap.add_argument("--post", type=float, default=1.5, help="post-roll (for RAM calc)")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    log = setup_logger("capture_soak", args.log)
    log.info("=== RAW capture soak test (%.0fs) ===", args.seconds)

    cam = RawMJPEGCamera(device=args.device, width=args.width, height=args.height,
                         fps=args.fps, exposure=args.exposure, gain=args.gain,
                         auto_exposure=args.auto_exposure, logger=log)
    if not cam.available:
        log.error("Raw path unavailable. `pip install linuxpy`. Aborting.")
        return 2
    if not cam.open():
        log.error("Camera open failed.")
        return 2

    # warm-up
    for _ in range(10):
        cam.read_raw()

    times = []
    sizes = []
    fails = 0
    last_check = time.time()
    quality = []          # (t, sharpness, brightness)

    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            ok, jpg = cam.read_raw()
            now = time.time()
            if ok and jpg:
                times.append(now)
                sizes.append(len(jpg))
                # periodic decode quality check
                if now - last_check >= args.check_every:
                    last_check = now
                    if _HAVE_CV2:
                        img = CameraV4L2.decode_jpeg(jpg, mono=True)
                        if img is not None:
                            sharp = cv2.Laplacian(img, cv2.CV_64F).var()
                            bright = float(img.mean())
                            quality.append((now - t0, sharp, bright))
                            log.info("  [%.0fs] fps(inst)~%.0f sharp=%.0f "
                                     "bright=%.1f jpg=%s",
                                     now - t0,
                                     1.0/statistics.mean(
                                         [b-a for a, b in zip(times[-30:], times[-29:])]
                                     ) if len(times) > 30 else 0,
                                     sharp, bright, human_bytes(len(jpg)))
            else:
                fails += 1
    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        cam.close()

    # ---- analysis ---------------------------------------------------------
    dur = time.time() - t0
    n = len(times)
    fps = n / dur if dur else 0
    deltas = [b - a for a, b in zip(times, times[1:])]
    mean_dt = statistics.mean(deltas) if deltas else 0
    std_dt = statistics.pstdev(deltas) if len(deltas) > 1 else 0
    worst_gap = max(deltas) if deltas else 0
    p99_gap = sorted(deltas)[int(len(deltas)*0.99)] if len(deltas) > 100 else worst_gap

    mean_sz = statistics.mean(sizes) if sizes else 0
    sz_cv = (statistics.pstdev(sizes) / mean_sz * 100) if mean_sz else 0

    # RAM projection for the ring buffer (raw JPEG bytes)
    ring_frames = int((args.pre + args.post) * fps)
    ring_ram = ring_frames * mean_sz

    log.info("---------------- SOAK SUMMARY ----------------")
    log.info("Duration            : %.1f s", dur)
    log.info("Frames              : %d", n)
    log.info("Sustained FPS       : %.1f (target %d)", fps, args.fps)
    log.info("Frame interval mean : %.2f ms", mean_dt*1000)
    log.info("Frame interval std  : %.2f ms  <-- jitter", std_dt*1000)
    log.info("Worst gap           : %.1f ms  <-- max blind moment", worst_gap*1000)
    log.info("99th-pct gap        : %.1f ms", p99_gap*1000)
    log.info("Read failures       : %d", fails)
    log.info("Mean JPEG size      : %s", human_bytes(mean_sz))
    log.info("JPEG size variation : %.1f%%  %s", sz_cv,
             "(stable -- good)" if sz_cv < 15 else
             "(HIGH -- auto-exposure may be active; set --exposure/--gain)")
    if quality:
        sharps = [q[1] for q in quality]
        brights = [q[2] for q in quality]
        log.info("Sharpness over run  : min %.0f / max %.0f %s",
                 min(sharps), max(sharps),
                 "(steady)" if (max(sharps)-min(sharps)) < 0.3*max(sharps)
                 else "(DRIFTING -- check focus lock)")
        log.info("Brightness over run : min %.1f / max %.1f", min(brights), max(brights))
    log.info("Ring buffer @ pre+post=%.1fs: ~%d frames, ~%s RAM",
             args.pre+args.post, ring_frames, human_bytes(ring_ram))
    log.info("----------------------------------------------")

    # verdict
    ok = True
    if fps < args.fps * 0.8:
        log.warning("VERDICT: FPS below 80%% of target -- investigate USB/CPU.")
        ok = False
    if worst_gap > 3 * mean_dt and mean_dt > 0:
        log.warning("VERDICT: worst gap is >3x mean interval -- occasional "
                    "stalls could drop a strike frame.")
        ok = False
    if sz_cv > 15 and not args.auto_exposure:
        log.warning("VERDICT: JPEG size unstable -- pin exposure with --exposure.")
        ok = False
    if ok:
        log.info("VERDICT: capture path looks solid for lightning duty.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
