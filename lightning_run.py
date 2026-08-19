#!/usr/bin/env python3
"""
lightning_run.py -- the always-on capture application.

Concept (from the project plan)
-------------------------------
        Lightning -> AS3935 IRQ -> Raspberry Pi
              camera frames stream continuously into a RAM ring buffer
              (holds the last PRE seconds)
        On IRQ: keep capturing for POST more seconds, then splice
              [pre-roll | post-roll] into one event and write it to USB.
        The live stream is NEVER continuously written to disk.

What gets recorded
------------------
* Events: each trigger saves a PRE+POST-second clip (default 1.5 + 1.5 = 3 s)
  plus a small JSON sidecar with the sensor reading and power state at trigger.
* Heartbeat: every --status-every seconds a row is appended to status_log.csv
  capturing FPS, ring depth, power/throttle flags, core volts, SoC temp,
  free space on the output drive, and event counts. This is the
  "what is going on during the run" record.
* On every saved event, the current sensor + power snapshot is written both
  into the event sidecar AND as a status row, so you always have the sensor
  data at the moment a picture/clip was taken.

Output formats
--------------
* --format mp4  : one video file per event (needs OpenCV VideoWriter + codec)
* --format npy  : raw frames stacked in a .npy (lossless, big) + sidecar
* --format png  : a folder of individual PNG frames per event
The MJPEG/mono raw-size math from the plan: 640x480 mono @120 FPS ~= 111 MB per
3 s event if stored raw; mp4/png are much smaller.

Run it
------
    python3 lightning_run.py --output /media/pi/USB/lightning_events
    python3 lightning_run.py            # auto-pick USB, else ./lightning_events

Stop with Ctrl-C -- it flushes and exits cleanly.
"""
import os
import sys
import json
import time
import signal
import argparse
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lc_common import (setup_logger, PowerMonitor, StatusCsv, resolve_output_dir,
                       find_usb_mounts, human_bytes, ts_compact, now_iso)   # noqa
from lc_camera import CameraV4L2, RingBuffer                                # noqa
from lc_sensor import AS3935Sensor                                          # noqa


class LightningRecorder:
    def __init__(self, args, log):
        self.args = args
        self.log = log
        self.stop = threading.Event()

        self.output_dir = resolve_output_dir(args.output, log)
        self.events_dir = os.path.join(self.output_dir, "events")
        os.makedirs(self.events_dir, exist_ok=True)

        self.status_csv = StatusCsv(os.path.join(self.output_dir, "status_log.csv"))
        self.power = PowerMonitor(logger=log)

        self.cam = CameraV4L2(device=args.device, width=args.width,
                              height=args.height, fps=args.fps,
                              fourcc=args.fourcc, mono=True, logger=log)
        self.ring = RingBuffer(pre_seconds=args.pre, fps=args.fps, logger=log)

        self.sensor = None
        if not args.no_sensor:
            self.sensor = AS3935Sensor(bus=args.bus, spi_bus=args.spi_bus,
                                       spi_dev=args.spi_dev, irq_gpio=args.irq_gpio,
                                       indoor=not args.outdoor, logger=log)

        # counters for the heartbeat
        self._frames_since_status = 0
        self._events_total = 0
        self._last_status = time.time()
        self._fps_window_start = time.time()

        # trigger coordination
        self._trigger_lock = threading.Lock()
        self._pending_trigger = None       # holds sensor event dict when armed

    # ------------------------------------------------------------------ #
    def start(self):
        if not self.cam.open():
            raise RuntimeError("Camera failed to open.")
        # apply manual exposure/gain if requested (stable brightness for motion)
        self._apply_camera_controls()
        # prime the pipeline
        for _ in range(5):
            self.cam.read()
        self.log.info("Camera live: %s", self.cam.describe())

        if self.sensor and self.sensor.available:
            self.sensor.start(self._on_sensor_event)
            if not self.sensor.comm_ok():
                self.log.warning("AS3935 not responding on the bus -- check "
                                 "wiring/SI strap/power. Triggers will not work.")
            # apply tuning from CLI args (each only if provided)
            a = self.args
            if getattr(a, "noise_floor", None) is not None:
                self.sensor.set_noise_floor(a.noise_floor)
                self.log.info("Sensor noise floor set to %d", a.noise_floor)
            if getattr(a, "watchdog", None) is not None:
                self.sensor.set_watchdog_threshold(a.watchdog)
                self.log.info("Sensor watchdog set to %d", a.watchdog)
            if getattr(a, "spike", None) is not None:
                self.sensor.set_spike_rejection(a.spike)
                self.log.info("Sensor spike rejection set to %d", a.spike)
            if getattr(a, "mask_disturbers", False):
                self.sensor.mask_disturbers(True)
                self.log.info("Sensor disturber masking ON (only lightning triggers).")
        elif self.sensor:
            self.log.warning("Sensor unavailable -- running camera-only "
                             "(manual SIGUSR1 trigger still works).")
            self.sensor = None

        self.log.info("Output dir: %s", self.output_dir)
        self._log_usb_state()

    def _apply_camera_controls(self):
        """Set manual exposure/gain via v4l2-ctl (stable brightness for motion)."""
        import subprocess
        a = self.args
        dev = a.device
        def setctrl(name, val):
            try:
                subprocess.run(["v4l2-ctl", "-d", dev, "--set-ctrl", f"{name}={val}"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=3)
            except Exception as e:
                self.log.warning("Could not set %s=%s: %s", name, val, e)
        if not getattr(a, "auto_exposure", False):
            setctrl("exposure_auto", 1)
            setctrl("auto_exposure", 1)
        if getattr(a, "exposure", None) is not None:
            setctrl("exposure_time_absolute", a.exposure)
            setctrl("exposure_absolute", a.exposure)
            self.log.info("Camera exposure set to %d (manual).", a.exposure)
        if getattr(a, "gain", None) is not None:
            setctrl("gain", a.gain)
            self.log.info("Camera gain set to %d.", a.gain)

    def _log_usb_state(self):
        usb = find_usb_mounts()
        if usb:
            for u in usb:
                self.log.info("USB: %s -> %s free %s / %s", u["device"],
                              u["mount"], human_bytes(u["free"]),
                              human_bytes(u["size"]))
        else:
            self.log.info("No removable USB drive detected.")

    # ------------------------------------------------------------------ #
    # sensor callback -- arm a trigger (runs in GPIO thread; keep it fast)
    # ------------------------------------------------------------------ #
    def _on_sensor_event(self, ev):
        # We only trigger a save on real lightning; disturbers/noise are logged.
        if ev["kind"] == "lightning" or self.args.trigger_on_disturber:
            with self._trigger_lock:
                if self._pending_trigger is None:
                    self._pending_trigger = ev
                    self.log.info("TRIGGER (%s) distance=%s energy=%d",
                                  ev["kind"], ev.get("distance_km"), ev["energy"])
        else:
            self.log.info("sensor: %s (not triggering) distance=%s",
                          ev["kind"], ev.get("distance_km"))

    # ------------------------------------------------------------------ #
    # main capture loop
    # ------------------------------------------------------------------ #
    def run(self):
        self.log.info("Capture loop started. Ctrl-C to stop.")
        try:
            while not self.stop.is_set():
                ok, frame = self.cam.read()
                now = time.time()
                if ok:
                    self.ring.push(now, frame)
                    self._frames_since_status += 1

                # a trigger armed? -> record the event
                trig = None
                with self._trigger_lock:
                    if self._pending_trigger is not None:
                        trig = self._pending_trigger
                        self._pending_trigger = None
                if trig is not None:
                    self._record_event(trig)

                # periodic heartbeat
                if now - self._last_status >= self.args.status_every:
                    self._write_status(reason="heartbeat")

                # manual key trigger (optional, for bench testing without storms)
                # handled via SIGUSR1 -> see signal handler
        except KeyboardInterrupt:
            self.log.info("Ctrl-C -- shutting down.")
        finally:
            self._shutdown()

    # ------------------------------------------------------------------ #
    def _record_event(self, trigger_ev):
        """Splice pre-roll from ring + capture POST seconds, then save."""
        t_trigger = time.time()
        pre = self.ring.snapshot_pre(t_trigger)
        self.log.info("Recording event: %d pre-roll frames, capturing %.1fs post...",
                      len(pre), self.args.post)

        post = []
        t_end = t_trigger + self.args.post
        while time.time() < t_end and not self.stop.is_set():
            ok, frame = self.cam.read()
            now = time.time()
            if ok:
                self.ring.push(now, frame)     # keep the ring fed
                post.append((now, frame))
                self._frames_since_status += 1

        frames = pre + post
        if not frames:
            self.log.warning("No frames captured for event -- skipping save.")
            return

        stamp = ts_compact()
        base = os.path.join(self.events_dir, f"event_{stamp}")
        power_snap = self.power.snapshot()

        try:
            saved_path = self._save_frames(base, frames)
        except Exception as e:
            self.log.error("Failed to save event: %s", e)
            return

        sidecar = {
            "event_id": stamp,
            "trigger_time": now_iso(),
            "n_frames": len(frames),
            "pre_frames": len(pre),
            "post_frames": len(post),
            "pre_seconds": self.args.pre,
            "post_seconds": self.args.post,
            "camera": self.cam.describe(),
            "sensor": {
                "kind": trigger_ev["kind"],
                "distance_km": trigger_ev.get("distance_km"),
                "energy": trigger_ev["energy"],
                "reason_code": trigger_ev["reason_code"],
            },
            "power": power_snap,
            "saved_as": os.path.basename(saved_path),
        }
        with open(base + ".json", "w") as f:
            json.dump(sidecar, f, indent=2)

        self._events_total += 1
        self.log.info("Saved event %s (%d frames) -> %s",
                      stamp, len(frames), saved_path)
        # also drop a status row tagged to this event, with sensor data attached
        self._write_status(reason=f"event:{stamp}", sensor_ev=trigger_ev)

    # ------------------------------------------------------------------ #
    def _save_frames(self, base, frames):
        fmt = self.args.format
        if fmt == "mp4":
            import cv2
            h, w = frames[0][1].shape[:2]
            path = base + ".mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # write at the *target* fps so playback speed is right
            vw = cv2.VideoWriter(path, fourcc, self.args.fps, (w, h), isColor=False)
            for _t, fr in frames:
                if fr.ndim == 3:
                    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                vw.write(fr)
            vw.release()
            return path
        elif fmt == "npy":
            import numpy as np
            path = base + ".npy"
            stack = np.stack([fr for _t, fr in frames])
            np.save(path, stack)
            # timestamps alongside
            with open(base + "_timestamps.json", "w") as f:
                json.dump([t for t, _fr in frames], f)
            return path
        elif fmt == "png":
            import cv2
            folder = base + "_frames"
            os.makedirs(folder, exist_ok=True)
            for i, (_t, fr) in enumerate(frames):
                cv2.imwrite(os.path.join(folder, f"f{i:04d}.png"), fr)
            return folder
        else:
            raise ValueError(f"unknown format {fmt}")

    # ------------------------------------------------------------------ #
    def _write_status(self, reason="heartbeat", sensor_ev=None):
        now = time.time()
        window = now - self._fps_window_start
        fps = self._frames_since_status / window if window > 0 else 0

        power_snap = self.power.snapshot()
        mask = self.power.get_throttled_raw()
        warnings = "|".join(self.power.active_warnings(mask)) if mask else ""

        usb = find_usb_mounts()
        free = usb[0]["free"] if usb else None

        row = {
            "time": now_iso(),
            "reason": reason,
            "fps_measured": round(fps, 2),
            "ring_frames": len(self.ring),
            "events_total": self._events_total,
            "throttled_hex": power_snap["throttled_hex"],
            "warnings": warnings,
            "core_volt_v": power_snap["core_volt_v"],
            "soc_temp_c": power_snap["soc_temp_c"],
            "usb_free_bytes": free,
            "usb_free_h": human_bytes(free),
            # sensor data at this moment (if an event triggered this row)
            "sensor_kind": sensor_ev["kind"] if sensor_ev else "",
            "sensor_distance_km": sensor_ev.get("distance_km") if sensor_ev else "",
            "sensor_energy": sensor_ev["energy"] if sensor_ev else "",
        }
        self.status_csv.write(row)

        msg = (f"status[{reason}] fps={row['fps_measured']} "
               f"ring={row['ring_frames']} events={row['events_total']} "
               f"volt={row['core_volt_v']} temp={row['soc_temp_c']} "
               f"usb_free={row['usb_free_h']}")
        if warnings:
            self.log.warning("%s  POWER: %s", msg, warnings)
        else:
            self.log.info(msg)

        self._frames_since_status = 0
        self._fps_window_start = now
        self._last_status = now

    # ------------------------------------------------------------------ #
    def manual_trigger(self):
        """Force a trigger (used by SIGUSR1) for bench testing."""
        with self._trigger_lock:
            if self._pending_trigger is None:
                self._pending_trigger = {
                    "kind": "manual", "distance_km": None,
                    "energy": 0, "reason_code": -1,
                }
                self.log.info("MANUAL trigger requested.")

    def _shutdown(self):
        self.stop.set()
        self._write_status(reason="shutdown")
        if self.sensor:
            self.sensor.stop()
        self.cam.close()
        self.log.info("Clean shutdown. Total events: %d", self._events_total)


# --------------------------------------------------------------------------- #
def build_argparser():
    ap = argparse.ArgumentParser(
        description="Always-on lightning camera recorder with RAM ring buffer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = ap.add_argument_group("camera")
    g.add_argument("--device", default="/dev/video0")
    g.add_argument("--width", type=int, default=640)
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--fps", type=int, default=120)
    g.add_argument("--fourcc", default="MJPG",
                   choices=["MJPG", "YUYV", "GREY", "Y8"])
    g.add_argument("--exposure", type=int, default=None,
                   help="manual exposure (v4l2 units); omit to leave as-is")
    g.add_argument("--gain", type=int, default=None, help="manual gain")
    g.add_argument("--auto-exposure", action="store_true",
                   help="allow auto-exposure (NOT recommended for motion capture)")

    g = ap.add_argument_group("event timing")
    g.add_argument("--pre", type=float, default=1.5,
                   help="seconds kept BEFORE the trigger (ring buffer depth)")
    g.add_argument("--post", type=float, default=1.5,
                   help="seconds captured AFTER the trigger")
    g.add_argument("--format", default="mp4", choices=["mp4", "npy", "png"],
                   help="how to store each event")
    g.add_argument("--recal-every", type=float, default=0,
                   help="auto-recalibrate exposure every N s during quiet "
                        "periods (0=off); never runs during an event")
    g.add_argument("--target-brightness", type=int, default=70,
                   help="sky-background brightness target for recalibration")

    g = ap.add_argument_group("sensor")
    g.add_argument("--no-sensor", action="store_true",
                   help="run without the AS3935 (camera-only / manual trigger)")
    g.add_argument("--bus", choices=["spi", "i2c"], default="spi")
    g.add_argument("--spi-bus", type=int, default=0)
    g.add_argument("--spi-dev", type=int, default=0)
    g.add_argument("--irq-gpio", type=int, default=17)
    g.add_argument("--outdoor", action="store_true",
                   help="outdoor AFE preset (recommended for rooftop/field)")
    g.add_argument("--noise-floor", type=int, default=None,
                   help="AS3935 noise floor 0..7 (higher rejects more noise)")
    g.add_argument("--watchdog", type=int, default=None,
                   help="AS3935 watchdog threshold 0..15")
    g.add_argument("--spike", type=int, default=None,
                   help="AS3935 spike rejection 0..15")
    g.add_argument("--mask-disturbers", action="store_true",
                   help="suppress disturber IRQs (only real lightning triggers)")
    g.add_argument("--trigger-on-disturber", action="store_true",
                   help="also save events on 'disturber' interrupts (debug)")

    g = ap.add_argument_group("output / logging")
    g.add_argument("--output", default=None,
                   help="output dir (default: auto-pick USB, else ./lightning_events)")
    g.add_argument("--status-every", type=float, default=120,
                   help="seconds between heartbeat status rows")
    g.add_argument("--log", default=None,
                   help="logfile path (default: <output>/run.log)")
    return ap


def main():
    args = build_argparser().parse_args()

    # default logfile lives next to the output
    out_preview = resolve_output_dir(args.output, None)
    logfile = args.log or os.path.join(out_preview, "run.log")
    log = setup_logger("lightning_run", logfile)

    log.info("========================================")
    log.info("Lightning camera recorder starting")
    log.info("pre=%.1fs post=%.1fs fps=%d res=%dx%d fourcc=%s format=%s",
             args.pre, args.post, args.fps, args.width, args.height,
             args.fourcc, args.format)
    log.info("========================================")

    rec = LightningRecorder(args, log)

    # allow `kill -USR1 <pid>` to force a test event without a real storm
    signal.signal(signal.SIGUSR1, lambda *_: rec.manual_trigger())

    try:
        rec.start()
        rec.run()
    except Exception as e:
        log.exception("Fatal error: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
