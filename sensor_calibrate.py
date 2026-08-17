#!/usr/bin/env python3
"""
sensor_calibrate.py -- sweep AS3935 tuning registers and record event rates.

The AS3935 has several registers that trade noise/disturber immunity against
sensitivity to real (especially distant) strikes:

  * noise floor (0..7)       -- how loud the RF background must be to raise a
                                "noise" interrupt
  * watchdog threshold (0..15) -- how strong a candidate must be to be examined
  * spike rejection (0..15)  -- how strictly the waveform shape is validated

Indoors, near a Pi and its PSU, disturbers/noise are unavoidable. The goal of
calibration is to find the LOWEST settings (best real-strike sensitivity) that
still keep the false-event rate acceptably low in your actual environment.

This tool sweeps combinations, listens for a fixed window at each, counts
events by category, and writes one CSV row per combination. Read the CSV to
pick your operating point: you want noise=0 and a low, steady disturber rate,
at the smallest thresholds that achieve it (so distant lightning still gets
through).

Because a real storm isn't usually available, this measures the FALSE-event
(noise + disturber) rate. Lower is better. Validate real sensitivity separately
with a piezo lighter (should still produce a disturber) once you've chosen a
setting.

Examples
--------
  # default sweep of noise floor 2..6, write CSV:
  python3 tests/sensor_calibrate.py --csv calib.csv

  # sweep noise floor AND spike rejection, 20 s each:
  python3 tests/sensor_calibrate.py --sweep-spike --dwell 20 --csv calib.csv

  # full grid (long!): noise x watchdog x spike
  python3 tests/sensor_calibrate.py --sweep-watchdog --sweep-spike --dwell 15 \
      --csv calib_full.csv
"""
import os
import sys
import time
import argparse
import threading
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lc_common import setup_logger, now_iso                # noqa: E402
from lc_sensor import AS3935Sensor                         # noqa: E402


def measure_window(sensor, dwell, log):
    """Listen for `dwell` seconds, return Counter of event kinds + energy stats."""
    counts = Counter()
    energies = []
    distances = []
    lock = threading.Lock()

    def on_event(ev):
        with lock:
            counts[ev["kind"]] += 1
            if ev["energy"]:
                energies.append(ev["energy"])
            if ev["distance_km"] is not None:
                distances.append(ev["distance_km"])

    # (re)bind the callback for this window
    sensor._callback = on_event
    t0 = time.time()
    while time.time() - t0 < dwell:
        time.sleep(0.2)
    dur = time.time() - t0

    with lock:
        total = sum(counts.values())
        false_events = counts.get("noise", 0) + counts.get("disturber", 0)
        row = {
            "noise": counts.get("noise", 0),
            "disturber": counts.get("disturber", 0),
            "lightning": counts.get("lightning", 0),
            "total_events": total,
            "false_per_min": round(false_events / dur * 60, 1),
            "noise_per_min": round(counts.get("noise", 0) / dur * 60, 1),
            "disturber_per_min": round(counts.get("disturber", 0) / dur * 60, 1),
            "mean_energy": round(sum(energies) / len(energies)) if energies else 0,
            "min_distance_km": min(distances) if distances else "",
        }
    return row


def main():
    ap = argparse.ArgumentParser(description="AS3935 calibration sweep")
    ap.add_argument("--bus", choices=["spi", "i2c"], default="spi")
    ap.add_argument("--spi-bus", type=int, default=0)
    ap.add_argument("--spi-dev", type=int, default=0)
    ap.add_argument("--irq-gpio", type=int, default=17)
    ap.add_argument("--i2c-addr", type=lambda x: int(x, 0), default=0x03)
    ap.add_argument("--indoor", action="store_true",
                    help="use indoor AFE preset (default is outdoor for calibration)")

    ap.add_argument("--dwell", type=float, default=15,
                    help="seconds to listen at each setting")
    ap.add_argument("--noise-min", type=int, default=2)
    ap.add_argument("--noise-max", type=int, default=6)
    ap.add_argument("--sweep-watchdog", action="store_true",
                    help="also sweep watchdog threshold (0,2,4,6)")
    ap.add_argument("--sweep-spike", action="store_true",
                    help="also sweep spike rejection (0,2,4,6)")
    ap.add_argument("--mask-disturbers", action="store_true",
                    help="mask disturbers during the sweep (measures noise only)")

    ap.add_argument("--csv", default="sensor_calib.csv")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    log = setup_logger("sensor_calib", args.log)
    log.info("=== AS3935 calibration sweep ===")

    sensor = AS3935Sensor(bus=args.bus, spi_bus=args.spi_bus, spi_dev=args.spi_dev,
                          i2c_addr=args.i2c_addr, irq_gpio=args.irq_gpio,
                          indoor=args.indoor, logger=log)
    if not sensor.available:
        log.error("Sensor backend unavailable. Aborting.")
        return 2

    # start once; we re-tune registers live between windows
    sensor.start(lambda ev: None)
    if not sensor.comm_ok():
        log.error("Sensor not responding on the bus -- fix wiring before "
                  "calibrating (see test_sensor.py).")
        sensor.stop()
        return 3

    if args.mask_disturbers:
        sensor.mask_disturbers(True)
        log.info("Disturbers masked -- measuring noise interrupts only.")

    noise_levels = list(range(args.noise_min, args.noise_max + 1))
    watchdog_levels = [0, 2, 4, 6] if args.sweep_watchdog else [None]
    spike_levels = [0, 2, 4, 6] if args.sweep_spike else [None]

    combos = [(n, w, s)
              for n in noise_levels
              for w in watchdog_levels
              for s in spike_levels]
    total_time = len(combos) * args.dwell
    log.info("%d combinations x %.0fs = ~%.0f s (%.1f min) total.",
             len(combos), args.dwell, total_time, total_time / 60)

    rows = []
    try:
        for i, (nf, wd, sr) in enumerate(combos, 1):
            sensor.set_noise_floor(nf)
            if wd is not None:
                sensor.set_watchdog_threshold(wd)
            if sr is not None:
                sensor.set_spike_rejection(sr)
            # let settings settle and clear any pending interrupt
            time.sleep(0.3)
            sensor.read_event()

            log.info("[%d/%d] noise=%d watchdog=%s spike=%s  listening %.0fs...",
                     i, len(combos), nf, wd, sr, args.dwell)
            stats = measure_window(sensor, args.dwell, log)
            row = {
                "time": now_iso(),
                "noise_floor": nf,
                "watchdog": "" if wd is None else wd,
                "spike_reject": "" if sr is None else sr,
                **stats,
            }
            rows.append(row)
            log.info("     -> noise=%d disturber=%d  false/min=%.1f  "
                     "mean_energy=%d",
                     stats["noise"], stats["disturber"],
                     stats["false_per_min"], stats["mean_energy"])
    except KeyboardInterrupt:
        log.info("Interrupted -- writing partial results.")
    finally:
        sensor.stop()

    # write CSV
    if rows:
        import csv
        fields = list(rows[0].keys())
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        log.info("Wrote %s (%d rows)", args.csv, len(rows))

        # print a compact ranked summary: quietest settings first
        ranked = sorted(rows, key=lambda r: r["false_per_min"])
        log.info("")
        log.info("QUIETEST SETTINGS (lowest false-event rate first):")
        log.info("%-6s %-9s %-6s %-12s %-10s", "noise", "watchdog", "spike",
                 "false/min", "mean_E")
        for r in ranked[:8]:
            log.info("%-6s %-9s %-6s %-12s %-10s",
                     r["noise_floor"], r["watchdog"], r["spike_reject"],
                     r["false_per_min"], r["mean_energy"])
        log.info("")
        log.info("Pick the SMALLEST thresholds that give an acceptable "
                 "false/min -- smaller = more sensitive to distant strikes. "
                 "Then confirm a piezo lighter still triggers a disturber.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
