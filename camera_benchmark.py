#!/usr/bin/env python3
"""
camera_benchmark.py -- sweep camera settings and report speed + image quality.

For each (resolution, fourcc, target-fps) combination it measures:
  * delivered FPS   : full cap.read() (grab + decode), what your app actually gets
  * grab-only FPS   : cap.grab() with no decode -> the raw USB/driver ceiling
  * decode cost     : delivered vs grab gap -> how much CPU JPEG-decode costs you
  * jitter (ms)     : std-dev of inter-frame intervals (lower = smoother)
  * read failures   : dropped/empty reads
  * sharpness       : variance of Laplacian (higher = sharper / better focus)
  * brightness      : mean pixel (0-255); flags over/under-exposure
  * frame bytes     : approx size of one decoded frame in RAM

Why grab-only matters: if grab-only FPS is high but delivered FPS is low, the
Pi's CPU (JPEG decode) is the bottleneck, not USB bandwidth. If both are low,
it's the USB link / driver / cabling.

Examples
--------
  # sweep the fast MJPG modes this OV9281 actually exposes:
  python3 tests/camera_benchmark.py --sweep

  # one specific setting, longer sample, save a sample frame:
  python3 tests/camera_benchmark.py --width 640 --height 400 \
      --fourcc MJPG --fps 240 --seconds 5 --save sample_640x400.png

  # write the full results table to CSV:
  python3 tests/camera_benchmark.py --sweep --csv bench.csv
"""
import os
import sys
import csv
import time
import argparse
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lc_common import setup_logger, human_bytes          # noqa: E402

import numpy as np                                        # noqa: E402
try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False


FOURCC = {"MJPG": "MJPG", "YUYV": "YUYV", "GREY": "GREY", "Y8": "Y8  "}

# The modes this OV9281 board actually advertises (from v4l2-ctl). Edit freely.
DEFAULT_SWEEP = [
    # (width, height, fourcc, target_fps)
    (160, 120, "MJPG", 240),
    (320, 240, "MJPG", 240),
    (640, 360, "MJPG", 240),
    (640, 400, "MJPG", 240),
    (1280, 720, "MJPG", 120),
    (1280, 800, "MJPG", 120),
    (640, 400, "YUYV", 30),
]


def open_cam(device, w, h, fourcc, fps):
    dev_index = int(device.replace("/dev/video", "")) if isinstance(device, str) \
        and device.startswith("/dev/video") else int(device)
    cap = cv2.VideoCapture(dev_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC.get(fourcc, "MJPG")))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def negotiated(cap):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f = cap.get(cv2.CAP_PROP_FPS)
    raw = int(cap.get(cv2.CAP_PROP_FOURCC))
    cc = "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4)).strip()
    return w, h, f, cc


def bench_one(device, w, h, fourcc, fps, seconds, log, save=None):
    cap = open_cam(device, w, h, fourcc, fps)
    if cap is None:
        log.error("  open failed for %dx%d %s", w, h, fourcc)
        return None

    nw, nh, nf, ncc = negotiated(cap)

    # warm-up
    for _ in range(8):
        cap.read()

    # ---- phase 1: full read (grab + decode) --------------------------------
    times, fails = [], 0
    sharp_samples, bright_samples = [], []
    last_frame = None
    t0 = time.time()
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        now = time.time()
        if ok and frame is not None:
            times.append(now)
            last_frame = frame
            # sample quality on ~every 10th frame to keep overhead low
            if len(times) % 10 == 0:
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                sharp_samples.append(cv2.Laplacian(g, cv2.CV_64F).var())
                bright_samples.append(float(g.mean()))
        else:
            fails += 1
    dur = time.time() - t0
    delivered = len(times) / dur if dur else 0
    deltas = [b - a for a, b in zip(times, times[1:])]
    jitter_ms = statistics.pstdev(deltas) * 1000 if len(deltas) > 1 else 0

    # ---- phase 2: grab-only (no decode) -> USB/driver ceiling --------------
    g0 = time.time()
    gn = 0
    while time.time() - g0 < min(2.0, seconds):
        if cap.grab():
            gn += 1
    grab_only = gn / (time.time() - g0)

    frame_bytes = last_frame.nbytes if last_frame is not None else 0
    sharp = statistics.mean(sharp_samples) if sharp_samples else 0
    bright = statistics.mean(bright_samples) if bright_samples else 0

    if save and last_frame is not None:
        cv2.imwrite(save, last_frame)
        log.info("  saved sample -> %s", save)

    cap.release()

    return {
        "req": f"{w}x{h}",
        "req_fourcc": fourcc,
        "req_fps": fps,
        "neg": f"{nw}x{nh}",
        "neg_fourcc": ncc,
        "neg_fps": round(nf, 1),
        "delivered_fps": round(delivered, 1),
        "grab_only_fps": round(grab_only, 1),
        "decode_penalty_fps": round(grab_only - delivered, 1),
        "jitter_ms": round(jitter_ms, 2),
        "read_fails": fails,
        "sharpness": round(sharp, 1),
        "brightness": round(bright, 1),
        "exposure_flag": ("DARK" if bright < 40 else
                          "BRIGHT" if bright > 215 else "ok"),
        "frame_bytes": frame_bytes,
        "frame_h": human_bytes(frame_bytes),
    }


def print_table(rows, log):
    cols = [
        ("req", "requested", 9),
        ("neg", "negotiated", 10),
        ("neg_fourcc", "fmt", 5),
        ("neg_fps", "modeFPS", 8),
        ("delivered_fps", "deliv", 6),
        ("grab_only_fps", "grab", 6),
        ("decode_penalty_fps", "decCost", 8),
        ("jitter_ms", "jit_ms", 7),
        ("read_fails", "fails", 6),
        ("sharpness", "sharp", 8),
        ("brightness", "bright", 7),
        ("exposure_flag", "exp", 7),
        ("frame_h", "ramSize", 8),
    ]
    header = "  ".join(f"{title:<{w}}" for _k, title, w in cols)
    log.info("")
    log.info("RESULTS")
    log.info(header)
    log.info("-" * len(header))
    for r in rows:
        line = "  ".join(f"{str(r.get(k, '')):<{w}}" for k, _t, w in cols)
        log.info(line)
    log.info("")
    log.info("deliv = FPS your app gets (grab+decode) | grab = USB ceiling (no "
             "decode)")
    log.info("decCost = FPS lost to JPEG decode on CPU | sharp = focus (higher "
             "better) | exp = exposure")


def bench_raw(device, w, h, fps, seconds, log):
    """Benchmark RAW MJPEG capture (no decode) via v4l2py, if available."""
    from lc_camera import RawMJPEGCamera
    cam = RawMJPEGCamera(device=device, width=w, height=h, fps=fps, logger=log)
    if not cam.available:
        log.warning("Raw path unavailable (install with: pip install v4l2py).")
        return None
    if not cam.open():
        log.error("Raw camera open failed.")
        return None
    for _ in range(8):
        cam.read_raw()
    res = cam.measure_fps(seconds)
    cam.close()
    log.info("  RAW MJPEG: %.1f fps (no decode) at %dx%d", res["fps"], w, h)
    return {"req": f"{w}x{h}", "mode": "RAW-MJPG", "delivered_fps": res["fps"],
            "read_fails": res["read_failures"]}


def main():
    ap = argparse.ArgumentParser(description="Camera speed + quality benchmark")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--sweep", action="store_true",
                    help="run the built-in mode sweep")
    ap.add_argument("--raw", action="store_true",
                    help="also benchmark RAW MJPEG capture (no decode) via v4l2py")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--fourcc", default="MJPG", choices=list(FOURCC))
    ap.add_argument("--fps", type=int, default=240)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--save", default=None, help="save one sample frame (single mode)")
    ap.add_argument("--csv", default=None, help="write results table to CSV")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    log = setup_logger("cam_bench", args.log)
    if not _HAVE_CV2:
        log.error("OpenCV not available in this venv.")
        return 2

    log.info("=== Camera benchmark (device %s) ===", args.device)

    if args.sweep:
        combos = DEFAULT_SWEEP
    else:
        combos = [(args.width, args.height, args.fourcc, args.fps)]

    rows = []
    for (w, h, cc, f) in combos:
        log.info("Testing %dx%d %s @%d for %.1fs ...", w, h, cc, f, args.seconds)
        save = args.save if not args.sweep else None
        r = bench_one(args.device, w, h, cc, f, args.seconds, log, save=save)
        if r:
            rows.append(r)
            log.info("  -> delivered %.1f fps (grab %.1f, decode cost %.1f), "
                     "jitter %.2f ms, sharp %.0f, %s",
                     r["delivered_fps"], r["grab_only_fps"],
                     r["decode_penalty_fps"], r["jitter_ms"],
                     r["sharpness"], r["exposure_flag"])
        # optional raw comparison for MJPG modes
        if args.raw and cc == "MJPG":
            bench_raw(args.device, w, h, f, args.seconds, log)
        time.sleep(0.3)

    print_table(rows, log)

    if args.csv and rows:
        with open(args.csv, "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
        log.info("Wrote %s", args.csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
