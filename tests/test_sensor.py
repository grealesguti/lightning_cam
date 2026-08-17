#!/usr/bin/env python3
"""
test_sensor.py -- verify the AS3935 lightning sensor.

What it does
------------
1. Opens the sensor (SPI by default, matching the wiring plan).
2. Configures + calibrates it.
3. Dumps a few key registers so you can confirm the bus actually works
   (all-0x00 or all-0xFF usually means wiring/CS is wrong).
4. Listens for IRQs and prints every event (noise / disturber / lightning)
   with estimated distance and energy.

You can trigger a *disturber* easily for testing by holding a sparking source
or a piezo lighter near the antenna, or by running a noisy motor nearby.

Examples
--------
    python3 tests/test_sensor.py                 # SPI0/CE0, IRQ on GPIO17
    python3 tests/test_sensor.py --bus i2c --irq-gpio 17
    python3 tests/test_sensor.py --outdoor --noise-floor 2 --seconds 120
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lc_common import setup_logger, now_iso           # noqa: E402
from lc_sensor import AS3935Sensor                     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="AS3935 sensor test")
    ap.add_argument("--bus", choices=["spi", "i2c"], default="spi")
    ap.add_argument("--spi-bus", type=int, default=0)
    ap.add_argument("--spi-dev", type=int, default=0, help="chip-select (CE0=0)")
    ap.add_argument("--irq-gpio", type=int, default=17, help="BCM pin for IRQ")
    ap.add_argument("--i2c-addr", type=lambda x: int(x, 0), default=0x03)
    ap.add_argument("--outdoor", action="store_true",
                    help="use outdoor AFE preset (less sensitive)")
    ap.add_argument("--noise-floor", type=int, default=None,
                    help="0..7, higher rejects more noise")
    ap.add_argument("--seconds", type=float, default=60,
                    help="how long to listen (0 = forever)")
    ap.add_argument("--log", default=None, help="optional logfile path")
    args = ap.parse_args()

    log = setup_logger("test_sensor", args.log)
    log.info("=== AS3935 sensor test ===")

    sensor = AS3935Sensor(
        bus=args.bus, spi_bus=args.spi_bus, spi_dev=args.spi_dev,
        i2c_addr=args.i2c_addr, irq_gpio=args.irq_gpio,
        indoor=not args.outdoor, logger=log)

    if not sensor.available:
        log.error("Sensor backend not available. Install spidev/RPi.GPIO and "
                  "run on the Pi. Aborting.")
        return 2

    count = {"lightning": 0, "disturber": 0, "noise": 0, "other": 0}

    def on_event(ev):
        k = ev["kind"] if ev["kind"] in count else "other"
        count[k] += 1
        dist = ev["distance_km"]
        dist_s = "out-of-range" if dist is None else f"{dist} km"
        log.info("[%s] %-10s distance=%s energy=%d",
                 now_iso(), ev["kind"], dist_s, ev["energy"])

    try:
        sensor.start(on_event)
        if args.noise_floor is not None:
            sensor.set_noise_floor(args.noise_floor)
            log.info("Noise floor set to %d", args.noise_floor)

        # Register sanity dump
        log.info("Register sanity check (INT=0x03, DIST=0x07):")
        log.info("  0x00=0x%02X  0x01=0x%02X  0x03=0x%02X  0x07=0x%02X",
                 sensor._read_reg(0x00), sensor._read_reg(0x01),
                 sensor._read_reg(0x03), sensor._read_reg(0x07))

        log.info("Listening for %s ...",
                 "ever" if args.seconds == 0 else f"{args.seconds:.0f}s")
        t0 = time.time()
        while args.seconds == 0 or time.time() - t0 < args.seconds:
            time.sleep(0.5)

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        sensor.stop()
        log.info("Summary: %s", count)

    return 0


if __name__ == "__main__":
    sys.exit(main())
