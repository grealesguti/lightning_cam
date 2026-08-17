#!/usr/bin/env python3
"""
pressure_test.py -- stress both subsystems together and watch for problems.

Why
---
Individually the camera and sensor may look fine, but under a sustained
high-FPS grab the Pi can:
  * drop frames (USB bandwidth / CPU),
  * undervolt / throttle (especially on battery),
  * miss sensor IRQs while the CPU is busy.

This script runs the camera flat-out AND the sensor listener AND the power
monitor at the same time, then reports:
  * sustained FPS + jitter,
  * frame-read failures,
  * how many sensor events landed during the load,
  * every undervoltage / throttle flag raised during the run,
  * min core voltage and max SoC temperature seen.

It is the go-to debugging tool: run it for a few minutes and read the summary.

Examples
--------
    python3 tests/pressure_test.py --seconds 120
    python3 tests/pressure_test.py --seconds 300 --fps 120 --fourcc GREY \
        --no-sensor       # camera+power only, if sensor not wired yet
"""
import os
import sys
import time
import argparse
import threading
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lc_common import setup_logger, PowerMonitor, now_iso, human_bytes  # noqa: E402
from lc_camera import CameraV4L2                                        # noqa: E402
from lc_sensor import AS3935Sensor                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Camera+sensor+power stress test")
    # camera
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=120)
    ap.add_argument("--fourcc", default="MJPG",
                    choices=["MJPG", "YUYV", "GREY", "Y8"])
    # sensor
    ap.add_argument("--no-sensor", action="store_true",
                    help="skip the sensor (camera+power only)")
    ap.add_argument("--bus", choices=["spi", "i2c"], default="spi")
    ap.add_argument("--irq-gpio", type=int, default=17)
    ap.add_argument("--outdoor", action="store_true")
    # run
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--power-every", type=float, default=5.0,
                    help="seconds between power snapshots")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    log = setup_logger("pressure", args.log)
    log.info("=== PRESSURE TEST (%.0fs) ===", args.seconds)

    # --- subsystems --------------------------------------------------------
    power = PowerMonitor(logger=log)
    cam = CameraV4L2(device=args.device, width=args.width, height=args.height,
                     fps=args.fps, fourcc=args.fourcc, mono=True, logger=log)
    if not cam.open():
        log.error("Camera failed to open. Aborting.")
        return 2

    sensor = None
    sensor_events = []
    if not args.no_sensor:
        sensor = AS3935Sensor(bus=args.bus, irq_gpio=args.irq_gpio,
                              indoor=not args.outdoor, logger=log)
        if sensor.available:
            sensor.start(lambda ev: sensor_events.append(ev))
        else:
            log.warning("Sensor unavailable -- continuing camera+power only.")
            sensor = None

    # --- shared state ------------------------------------------------------
    stop = threading.Event()
    frame_times = []
    read_failures = 0
    volt_min = [None]
    temp_max = [None]
    warnings_seen = set()

    # --- power monitor thread ---------------------------------------------
    def power_loop():
        while not stop.is_set():
            snap = power.snapshot()
            mask = power.get_throttled_raw()
            if mask:
                for w in power.active_warnings(mask):
                    if w not in warnings_seen:
                        warnings_seen.add(w)
                        log.warning("POWER: %s (throttled=%s)",
                                    w, snap["throttled_hex"])
            v = snap["core_volt_v"]
            t = snap["soc_temp_c"]
            if v is not None:
                volt_min[0] = v if volt_min[0] is None else min(volt_min[0], v)
            if t is not None:
                temp_max[0] = t if temp_max[0] is None else max(temp_max[0], t)
            log.info("power: %s  volt=%s temp=%s", snap["throttled_hex"],
                     v, t)
            stop.wait(args.power_every)

    pw = threading.Thread(target=power_loop, daemon=True)
    pw.start()

    # --- camera hammer loop (main thread) ---------------------------------
    log.info("Hammering camera at target %d FPS ...", args.fps)
    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            ok, _frame = cam.read()
            now = time.time()
            if ok:
                frame_times.append(now)
            else:
                read_failures += 1
    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        stop.set()
        pw.join(timeout=2)
        if sensor:
            sensor.stop()
        cam.close()

    # --- analysis ----------------------------------------------------------
    dur = time.time() - t0
    n = len(frame_times)
    fps = n / dur if dur else 0

    # inter-frame jitter
    deltas = [b - a for a, b in zip(frame_times, frame_times[1:])]
    jitter_ms = statistics.pstdev(deltas) * 1000 if len(deltas) > 1 else 0
    mean_dt_ms = statistics.mean(deltas) * 1000 if deltas else 0

    log.info("---------------- SUMMARY ----------------")
    log.info("Duration           : %.1f s", dur)
    log.info("Frames captured    : %d", n)
    log.info("Sustained FPS      : %.2f (target %d)", fps, args.fps)
    log.info("Mean frame interval: %.2f ms  (jitter s.d. %.2f ms)",
             mean_dt_ms, jitter_ms)
    log.info("Read failures      : %d", read_failures)
    if sensor is not None:
        kinds = {}
        for e in sensor_events:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        log.info("Sensor events      : %d %s", len(sensor_events), kinds)
    log.info("Min core voltage   : %s V", volt_min[0])
    log.info("Max SoC temp       : %s C", temp_max[0])
    if warnings_seen:
        log.warning("POWER WARNINGS during run: %s", sorted(warnings_seen))
    else:
        log.info("Power warnings     : none (clean run)")
    log.info("-----------------------------------------")

    # exit code reflects health: nonzero if throttled or badly under target
    if warnings_seen or fps < args.fps * 0.8:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
