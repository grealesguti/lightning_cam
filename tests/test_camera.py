#!/usr/bin/env python3
"""
test_camera.py -- verify the OV9281 USB camera and its real throughput.

What it does
------------
1. Lists the camera's actual V4L2 modes (runs `v4l2-ctl --list-formats-ext`
   for you if available) so you can see which width/height/FPS combos exist.
2. Opens the camera at the requested resolution / FPS / format.
3. Reports the *negotiated* format (what the driver actually gave you --
   often not what you asked for).
4. Measures the real achieved FPS over N seconds.
5. Optionally saves one sample frame so you can eyeball focus/exposure.

Examples
--------
    python3 tests/test_camera.py --list                    # just show modes
    python3 tests/test_camera.py --width 640 --height 480 --fps 120
    python3 tests/test_camera.py --fourcc GREY --save sample.png
"""
import os
import sys
import time
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lc_common import setup_logger                     # noqa: E402
from lc_camera import CameraV4L2                        # noqa: E402


def list_v4l2_modes(device, log):
    """Run v4l2-ctl to enumerate real camera modes."""
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "--list-formats-ext", "-d", device],
            stderr=subprocess.STDOUT, timeout=10).decode()
        log.info("v4l2-ctl modes for %s:\n%s", device, out)
    except FileNotFoundError:
        log.warning("v4l2-ctl not found. Install with: sudo apt install v4l-utils")
    except Exception as e:
        log.warning("Could not list modes: %s", e)


def main():
    ap = argparse.ArgumentParser(description="OV9281 camera test")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=120)
    ap.add_argument("--fourcc", default="MJPG",
                    choices=["MJPG", "YUYV", "GREY", "Y8"])
    ap.add_argument("--mono", action="store_true", default=True)
    ap.add_argument("--color", dest="mono", action="store_false")
    ap.add_argument("--seconds", type=float, default=3.0,
                    help="duration of the FPS measurement")
    ap.add_argument("--list", action="store_true",
                    help="only list V4L2 modes and exit")
    ap.add_argument("--save", default=None, help="save one sample frame to this path")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    log = setup_logger("test_camera", args.log)
    log.info("=== OV9281 camera test ===")

    list_v4l2_modes(args.device, log)
    if args.list:
        return 0

    cam = CameraV4L2(device=args.device, width=args.width, height=args.height,
                     fps=args.fps, fourcc=args.fourcc, mono=args.mono, logger=log)
    if not cam.open():
        log.error("Camera open failed. Aborting.")
        return 2

    try:
        # warm-up (first frames are often slow while the pipeline fills)
        for _ in range(5):
            cam.read()

        log.info("Negotiated format: %s", cam.describe())
        log.info("Measuring real FPS for %.1fs ...", args.seconds)
        res = cam.measure_fps(args.seconds)
        log.info("RESULT: %d frames in %.3fs -> %.2f FPS (read failures: %d)",
                 res["frames"], res["seconds"], res["fps"], res["read_failures"])

        target = args.fps
        if res["fps"] < target * 0.8:
            log.warning("Achieved FPS is well below target (%d). Try: lower "
                        "resolution, MJPG format, a USB3 port, or check "
                        "`v4l2-ctl` modes above.", target)

        if args.save:
            ok, frame = cam.read()
            if ok:
                import cv2
                cv2.imwrite(args.save, frame)
                log.info("Saved sample frame -> %s (shape %s)",
                         args.save, getattr(frame, "shape", "?"))
            else:
                log.warning("Could not grab sample frame.")
    finally:
        cam.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
