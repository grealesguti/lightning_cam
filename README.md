# Lightning Camera — OV9281 + AS3935 on Raspberry Pi

A trigger-based recorder: an **AS3935 RF lightning sensor** fires an interrupt,
and an **OV9281 global-shutter USB camera** saves a short clip spanning the
strike (pre-roll from a RAM ring buffer + post-roll). The live stream is
**never** continuously written to disk — only triggered events land on the USB
drive. Post-processing tools then reconstruct the strike's **path** and its
**motion over time**.

```
   Lightning → AS3935 (IRQ) → Raspberry Pi
                                   │
         camera frames ──► RAM ring buffer (last PRE seconds)
                                   │  on trigger
                          keep capturing POST seconds
                                   │
                        [pre-roll | post-roll] = one event
                                   │
                             External USB drive
                                   │
                     offline: path + motion analysis
```

---

## Table of contents
1. [Files](#1-files)
2. [Install](#2-install)
3. [Wiring](#3-wiring)
4. [Find your USB drive](#4-find-your-usb-drive)
5. [Testing workflow](#5-testing-workflow)
6. [Camera: speed & quality](#6-camera-speed--quality)
7. [Sensor: calibration & tuning](#7-sensor-calibration--tuning)
8. [Running the recorder](#8-running-the-recorder)
8b. [Web GUI control panel](#8b-web-gui-control-panel)
9. [Analysis: path & motion](#9-analysis-path--motion)
10. [Power / battery monitoring](#10-power--battery-monitoring)
11. [Run at boot](#11-run-at-boot)
12. [Storage math](#12-storage-math)
13. [Troubleshooting](#13-troubleshooting)
14. [Future hardware ideas](#14-future-hardware-ideas)

---

## 1. Files

| File | Purpose |
|------|---------|
| `lc_common.py` | Shared helpers: power monitor, USB discovery, logging, CSV status. |
| `lc_sensor.py` | `AS3935Sensor` driver — SPI/I²C, lgpio IRQ, noise/watchdog/spike/mask tuning, comms verification. |
| `lc_camera.py` | `CameraV4L2` (decoded), `RawMJPEGCamera` (raw no-decode + exposure/gain + auto-recal), `RingBuffer`. |
| `lightning_run.py` | The always-on recorder (ring buffer + triggered saves + heartbeat log). |
| `lightning_gui.py` | Web control panel — SSH-friendly live preview, settings, sensor/power view, detach-and-keep-running. |
| **Camera tests** | |
| `tests/test_camera.py` | Verify the camera, list V4L2 modes, measure real FPS. |
| `tests/camera_benchmark.py` | Sweep resolutions/formats; report speed + image quality; `--raw` compares the no-decode path. |
| `tests/capture_soak.py` | Long soak test of the raw path: sustained FPS, frame-gap jitter, exposure stability, auto-recal, RAM budget. |
| **Sensor tests** | |
| `tests/test_sensor.py` | Verify the sensor, dump registers, print live events. |
| `tests/sensor_calibrate.py` | Sweep noise-floor/watchdog/spike; log false-event rates to CSV to pick a setting. |
| **Combined / analysis** | |
| `tests/pressure_test.py` | Stress camera + sensor + power together and report health. |
| `tests/lightning_path.py` | Reconstruct the final channel PATH (max-stack / diff-stack / mask). |
| `tests/lightning_motion.py` | Reconstruct MOTION over time (colour-by-time, centroid vector, optical flow, replay). |

---

## 2. Install

Recent Raspberry Pi OS (Bookworm) is **externally managed** — system-wide
`pip3 install` is blocked (`error: externally-managed-environment`). Use a
virtual environment.

```bash
cd ~/lightning_cam

sudo apt update
sudo apt install -y python3-full python3-venv v4l-utils i2c-tools

python3 -m venv --system-site-packages venv
source venv/bin/activate

pip install --upgrade pip
pip install opencv-python numpy spidev lgpio smbus2 linuxpy
```

Package notes:
- **`lgpio`** — GPIO for the sensor IRQ. The old `RPi.GPIO` fails on Bookworm
  with `RuntimeError: Failed to add edge detection` because the sysfs GPIO
  interface was removed; `lgpio` uses the modern `gpiochip` character device.
  The driver falls back to `RPi.GPIO` automatically on older systems.
- **`linuxpy`** — the raw-MJPEG (no-decode) capture path. Enables full-frame-rate
  capture (see §6). If absent, the code falls back to the decoded `CameraV4L2`.
- **`opencv-python` slow to build on a Pi 3B+?** Use the apt build instead:
  ```bash
  deactivate
  sudo apt install -y python3-opencv
  cd ~/lightning_cam && rm -rf venv
  python3 -m venv --system-site-packages venv && source venv/bin/activate
  pip install numpy spidev lgpio smbus2 linuxpy    # cv2 comes from system
  python -c "import cv2; print('cv2', cv2.__version__)"
  ```

**Activate the venv every session** before running anything:
```bash
cd ~/lightning_cam && source venv/bin/activate
```

### Enable the buses
```bash
sudo raspi-config    # Interface Options → enable SPI (and/or I2C)
ls /dev/spidev*      # e.g. /dev/spidev0.0  /dev/spidev0.1
ls /dev/i2c*         # e.g. /dev/i2c-1
```
`spidev0.0` = chip-select **CE0** (`--spi-dev 0`); `spidev0.1` = **CE1**
(`--spi-dev 1`).

---

## 3. Wiring

The AS3935's `SI` pin selects the interface: **SI = high → SPI, SI = low →
I²C** (verify against your board's silkscreen — some invert this). SI must be
tied to a fixed level, not a GPIO. If your board has an `EN` pin, it usually
needs pulling **high** to power the chip.

SPI wiring:

| AS3935 | Raspberry Pi | BCM | Physical pin |
|--------|--------------|-----|--------------|
| VCC | 3.3 V | — | pin 1 |
| GND | GND | — | pin 6 |
| SI | 3.3 V (selects SPI) | — | pin 17 |
| CS | SPI0_CE0 | GPIO 8 | pin 24 |
| MISO | SPI0_MISO | GPIO 9 | pin 21 |
| MOSI | SPI0_MOSI | GPIO 10 | pin 19 |
| SCL/SCLK | SPI0_SCLK | GPIO 11 | pin 23 |
| IRQ | GPIO 17 | GPIO 17 | pin 11 |

> ⚠️ **"17" is ambiguous.** IRQ goes to **GPIO 17 = physical pin 11**. `SI` to
> 3.3 V uses **physical pin 17** (a power pin). Don't confuse the two.
>
> ⚠️ **Verify VCC = 3.3 V, not 5 V** — many GY-AS3935 boards are 3.3 V only.
>
> ⚠️ If the register dump in `test_sensor.py` reads all `0x00`/`0xFF`, the bus
> isn't communicating — recheck CS, SI strap, power, and solder joints. The
> driver now reports `AS3935 NOT RESPONDING` explicitly in this case.

**Prefer I²C?** Tie SI low and use `--bus i2c`. Confirm the chip appears:
```bash
i2cdetect -y 1        # expect a device at 0x03
```

---

## 4. Find your USB drive

```bash
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT,RM,LABEL   # RM=1 = removable
findmnt -t vfat,exfat,ext4 -o SOURCE,TARGET,FSTYPE,SIZE
dmesg | tail -n 20                              # what was just plugged in
ls -l /dev/disk/by-label/                       # stable name symlinks
df -h /media/pi/*                               # free space
```

Mount manually if needed:
```bash
sudo mkdir -p /media/pi/USB
sudo mount /dev/sda1 /media/pi/USB
# exFAT: sudo apt install exfat-fuse exfatprogs
```

The scripts **auto-detect** a removable mount if you omit `--output`. See what
they'd pick:
```bash
python3 -c "from lc_common import find_usb_mounts,human_bytes; \
[print(u['device'],u['mount'],human_bytes(u['free'])) for u in find_usb_mounts()]"
```

---

## 5. Testing workflow

Do these in order. Each step gates the next.

**Step 1 — camera modes & real FPS**
```bash
v4l2-ctl --list-formats-ext -d /dev/video0     # what the camera actually offers
python3 tests/test_camera.py --width 640 --height 400 --fps 120 --fourcc MJPG
```

**Step 2 — camera benchmark (speed + quality + raw path)**
```bash
python3 tests/camera_benchmark.py --sweep --csv bench.csv
python3 tests/camera_benchmark.py --raw --width 640 --height 400 --fps 120 --seconds 5
```
See §6 for how to read the table.

**Step 3 — camera soak test (before trusting long runs)**
```bash
python3 tests/capture_soak.py --seconds 300 --exposure 200 --gain 20
```
See §6.

**Step 4 — sensor bus check & live events**
```bash
python3 tests/test_sensor.py --seconds 30                 # SPI0/CE0, IRQ GPIO17
python3 tests/test_sensor.py --bus i2c --i2c-addr 0x03    # I²C alternative
```
Register dump should be **non-zero**. Click a piezo BBQ lighter ~10–30 cm from
the antenna → should log a `disturber`, proving the IRQ path end-to-end.

**Step 5 — sensor calibration sweep**
```bash
python3 tests/sensor_calibrate.py --csv calib.csv
```
See §7.

**Step 6 — combined stress test**
```bash
python3 tests/pressure_test.py --seconds 120
```
Runs camera + sensor + power together; reports sustained FPS, jitter, dropped
reads, sensor hits, and every undervoltage/throttle flag. Exit code is non-zero
if the Pi throttled or FPS fell well below target.

---

## 6. Camera: speed & quality

### What lightning needs
The luminous channel glows/flickers over ~10–100 ms. To see **how it moved**
you need several frames across that window:

| Frames across the glow | FPS | What you get |
|------------------------|-----|--------------|
| 1–2 | ≤30 | that a strike happened |
| 3–5 | ~60 | rough path |
| **5–15** | **120–240** | **direction & motion** ← the goal |

### The decode bottleneck (and the fix)
On a Pi 3B+, OpenCV's `cap.read()` JPEG-decodes every frame on the CPU. Typical
measured result at 640×400 MJPG:

| Path | FPS | Bottleneck |
|------|-----|------------|
| decoded (`CameraV4L2`) | ~50 | CPU JPEG decode |
| **raw (`RawMJPEGCamera`)** | **~238** | none — decode skipped |

`camera_benchmark.py` columns: `deliv` = FPS your app gets (grab+decode);
`grab` = USB ceiling (no decode); `decCost` = FPS lost to decode; `sharp` =
focus (variance of Laplacian, higher = sharper); `bright` = mean pixel + an
exposure flag; `ramSize` = decoded frame size in RAM.

The **raw path stores compressed JPEG bytes** in the ring buffer and decodes
only the frames you keep (at save/review time via
`CameraV4L2.decode_jpeg()`). This is the recommended capture path — full frame
rate *and* ~1/10th the RAM, so you keep resolution **and** speed (important for
distant/faint strikes, where resolution matters more than raw FPS).

### Ensuring quality during capture
Three things, in priority order:

1. **Manual exposure — always.** Auto-exposure "pumps" brightness frame to
   frame and **corrupts motion analysis**. Set it manually:
   ```bash
   v4l2-ctl -d /dev/video0 --list-ctrls        # find exposure/gain controls
   python3 tests/capture_soak.py --exposure 200 --gain 20
   ```
   `RawMJPEGCamera(exposure=…, gain=…, auto_exposure=False)` disables
   auto-exposure and pins your values.

2. **Lock focus at infinity.** For sky, focus at infinity and tape the ring so
   it can't drift. The soak test's periodic sharpness check confirms it holds.

3. **Consistent frame timing.** The soak test reports frame-interval std-dev and
   **worst gap** — the largest "blind moment" where a strike could fall between
   frames. This is the single most important number for motion capture.

### Auto-recalibration (optional)
Ambient light drifts over a night (dusk, moonrise, cloud). `RawMJPEGCamera`
can **periodically re-tune exposure** toward a target sky-background brightness,
**without** the frame-to-frame pumping of auto-exposure: it measures the sky
during quiet periods and nudges exposure by a fraction of the error, so the
setting stays fixed while a strike is captured.

```bash
# recalibrate exposure every 60 s toward a sky brightness of 70/255:
python3 tests/capture_soak.py --seconds 600 --recal-every 60 --target-brightness 70
```
In code:
```python
cam = RawMJPEGCamera(device="/dev/video0", width=640, height=400, fps=120,
                     exposure=200, gain=20, auto_exposure=False, logger=log)
cam.open()
# ...in your loop, only during quiet periods (never mid-event):
if now - last_recal >= recal_seconds:
    cam.recalibrate(target_brightness=70)   # damped; converges then holds
    last_recal = now
```
`recalibrate()` uses the median of a decoded sample frame (so an in-frame bright
object doesn't skew it), moves a fraction of the proportional error to avoid
oscillation, and returns a dict describing what it changed. **Call it only
between events**, so exposure is constant across any captured strike.

### The soak test
`tests/capture_soak.py` runs the raw path the way the recorder will, for
minutes, and reports:
- sustained FPS over the whole run (not a 3 s burst)
- frame-interval **mean / std / worst gap / 99th-pct gap**
- read failures over time
- **JPEG-size variation** — a proxy for exposure stability; >15% means
  auto-exposure is still active and will corrupt motion analysis
- periodic decode checks: sharpness + brightness, to catch focus/exposure drift
- projected **ring-buffer RAM** at your `--pre`/`--post` and measured frame size
- optional `--recal-every` auto-recalibration
- a pass/fail **verdict**

```bash
python3 tests/capture_soak.py --seconds 300 --exposure 200 --gain 20 \
    --pre 1.5 --post 1.5 --recal-every 120
```

---

## 7. Sensor: calibration & tuning

The AS3935 trades noise/disturber immunity against sensitivity to real
(especially distant) strikes. `lc_sensor.py` exposes:

| Method | Register | Effect |
|--------|----------|--------|
| `set_noise_floor(0..7)` | 0x01 | RF background level to raise a "noise" IRQ |
| `set_watchdog_threshold(0..15)` | 0x01 | candidate strength gate; higher = more spike-immune |
| `set_spike_rejection(0..15)` | 0x02 | waveform-shape validation; higher rejects disturbers |
| `set_min_strikes(1/5/9/16)` | 0x02 | strikes before an IRQ fires; suppresses isolated triggers |
| `mask_disturbers(bool)` | 0x03 | suppress disturber IRQs entirely (recommended for the recorder) |

### Calibration sweep
```bash
python3 tests/sensor_calibrate.py --csv calib.csv                 # noise floor 2..6
python3 tests/sensor_calibrate.py --sweep-spike --dwell 20 --csv calib.csv
python3 tests/sensor_calibrate.py --sweep-watchdog --sweep-spike --dwell 15 --csv calib_full.csv
```
It holds the sensor open, re-tunes registers live between windows, counts events
per category over each dwell, and writes one CSV row per combination
(`noise_floor, watchdog, spike_reject, noise, disturber, lightning,
false_per_min, …, mean_energy, min_distance_km`). It prints a ranked summary,
quietest first.

**How to choose:** pick the **smallest** thresholds that give an acceptable
`false_per_min` — smaller = more sensitive to distant strikes. Then confirm a
piezo lighter still triggers a disturber (a config with zero events might be
deaf, not clean).

> **Interference check:** if the false rate barely responds to the noise floor
> **and** disturbers show a near-constant `mean_energy`, you likely have a
> periodic interferer nearby — commonly the Pi, its switching PSU, or the USB
> camera (the AS3935 is a sensitive ~500 kHz receiver). No register fixes this.
> **Move the sensor 30–50 cm away** from the Pi and USB cable, power the Pi from
> a battery/linear supply, and re-run the sweep. Calibrate where the sensor will
> actually live — outdoors away from electronics is far quieter.

---

## 8. Running the recorder

```bash
# auto-pick USB, defaults 1.5s pre + 1.5s post, 640x480@120 MJPG, mp4
python3 lightning_run.py

# explicit USB, raw frames, outdoor sensor preset, status every 60 s
python3 lightning_run.py --output /media/pi/USB/lightning_events \
    --format npy --outdoor --status-every 60

# bench test without a storm: camera-only + manual trigger
python3 lightning_run.py --no-sensor &
kill -USR1 $!        # forces one event to be recorded
```

### Key arguments (`--help` for the full list)

| Arg | Default | Meaning |
|-----|---------|---------|
| `--pre` / `--post` | 1.5 / 1.5 | seconds before/after the trigger |
| `--fps` | 120 | target capture rate |
| `--width/--height` | 640/480 | resolution (verify real modes with `v4l2-ctl`) |
| `--fourcc` | MJPG | MJPG / YUYV / GREY / Y8 |
| `--format` | mp4 | `mp4` (small), `npy` (raw lossless + timestamps), `png` (per-frame) |
| `--output` | auto | output dir; auto-picks USB if omitted |
| `--status-every` | 120 | seconds between heartbeat status rows |
| `--no-sensor` | off | run without AS3935 (manual/`SIGUSR1` trigger) |
| `--outdoor` | off | less-sensitive AFE preset for rooftop/field |
| `--trigger-on-disturber` | off | also save on disturber IRQs (debug) |

> **For motion analysis, prefer `--format npy` or `png`**, which preserve
> individual frames (and, for npy, a `_timestamps.json` sidecar). `mp4` re-times
> and re-compresses frames and is not ideal for the motion tools.

### What lands on the drive
Per event under `<output>/events/`:
- `event_<timestamp>.mp4` / `.npy` (+ `_timestamps.json`) / `_frames/` PNGs
- `event_<timestamp>.json` — sensor reading (kind, distance, energy) **and**
  power/throttle state at the trigger moment

At the output root:
- `status_log.csv` — periodic "what's going on" record (see §10)
- `run.log` — full rotating text log

---

## 8b. Web GUI control panel

`lightning_gui.py` is a browser-based control panel that works **over SSH**
(no X forwarding) from a laptop or phone. It shows a live preview, lets you set
every option, displays sensor and power data in real time, lets you pick the
output USB from a dropdown, and can **detach** — stop the web UI but leave the
recorder running in the background with the current settings.

```bash
# local-only (safest); reach it via an SSH tunnel:
python3 lightning_gui.py
#   then on your laptop:  ssh -L 8080:localhost:8080 eclipse3@<pi-ip>
#   and open http://localhost:8080

# or bind to the LAN (no auth — only on a trusted network):
python3 lightning_gui.py --host 0.0.0.0 --port 8080
#   open http://<pi-ip>:8080
```

Panels:
- **Live preview** — low-rate MJPEG for framing and focus (not the capture rate).
- **Sensor data** — event type, storm distance (km), relative energy, live counts,
  and the last events. (What the AS3935 can/can't measure — see below.)
- **Power / health** — throttle flags, core volts, SoC temp, events saved.
- **Capture settings** — resolution, fps, exposure, gain, pre/post, format,
  and the recalibration interval + target brightness.
- **Sensor settings** — bus, IRQ pin, noise floor, watchdog, spike, mask
  disturbers, outdoor preset.
- **Output USB** — dropdown of detected removable drives (blank = auto-pick),
  with a Rescan button.
- **Control** — Apply settings, Start/Stop recording, and **Detach GUI (keep
  recording)**.

**Event-safe recalibration:** when auto-recalibration is on, it re-tunes
exposure only while **no event is active**. The recorder sets an `event_active`
flag around each capture, and `RawMJPEGCamera.recalibrate(event_active=…)`
hard-skips if that flag is set — so exposure is guaranteed constant across any
captured strike.

> Security: the panel has no authentication. Default bind is `127.0.0.1`
> (local-only); use the SSH tunnel above for remote access, or only bind
> `0.0.0.0` on a trusted network.

### What the AS3935 sensor can measure (and how)
The AS3935 is an **RF receiver, not a light sensor**. A tuned ~500 kHz loop
antenna listens for the electromagnetic signature of a discharge; it never sees
the flash optically. From each event it provides:
- **Event type** (register 0x03): lightning, disturber (man-made interference),
  or noise-too-high.
- **Storm distance** (register 0x07): a coarse 1–40 km estimate to the *leading
  edge of the storm* (not a single bolt), read from a calibrated table in
  silicon based on how the RF signature attenuates. Expect kilometre-scale
  buckets and an "out of range" code.
- **Relative energy** (registers 0x04–0x06): a 20-bit dimensionless number the
  distance algorithm uses internally — **not** joules; useful only for
  comparing events.

It has **no optical sensing and no direction/bearing**. Direction and the visual
channel are what the camera provides.

---

## 9. Analysis: path & motion

Run these **offline** on saved events (`.npy` stack or a folder of png/jpg
frames). Heavy processing is deferred here so capture stays fast.

### Final path — `tests/lightning_path.py`
Collapse an event into one image of the channel.
```bash
python3 tests/lightning_path.py --input event.npy --method max  --out path.png --report
python3 tests/lightning_path.py --input event.npy --method diff --out path.png --mask skel.png
```
- `max` — per-pixel brightest across frames; shows the full channel even if it
  flickered across frames.
- `diff` — subtracts a median sky baseline first, removing static lights/horizon
  so only the transient channel remains.
- `--mask` — threshold + thin to a skeleton trace of the path.
- `--report` — which frames actually contain the strike.

### Motion over time — `tests/lightning_motion.py`
Preserve the **time dimension**: direction and development, not just the final
shape.

| Flag | View |
|------|------|
| `--color-time OUT.png` | one image, channel tinted **blue→red by time** (best single "direction" still) |
| `--animate OUT.mp4` | slowed replay of just the strike frames |
| `--timeline OUT.png` | contact sheet of strike frames in order |
| `--per-frame-new DIR` | each frame's **newly-lit** pixels (growth increment by increment) |
| `--centroid OUT.png` | tracks brightness centroid; reports **direction + speed**; draws arrows |
| `--flow DIR` | dense optical-flow arrow overlays between consecutive frames |

```bash
# direction-of-development still + slowed replay:
python3 tests/lightning_motion.py --input event.npy --color-time evo.png --animate replay.mp4 --report

# quantify direction & speed (uses per-frame timestamps if present):
python3 tests/lightning_motion.py --input event.npy --centroid track.png
```
`--all-frames` skips strike detection; `--z` tunes detection sensitivity.

> **Physics note:** a single return stroke propagates in microseconds — no frame
> rate freezes the leader itself. What these tools show is the frame-to-frame
> evolution of the luminous channel (successive strokes, branch development,
> fade), which is still very informative about direction.

---

## 10. Power / battery monitoring

**Can we record the Pi's power status over time on battery? Yes.** The Pi can't
report input voltage directly, but firmware exposes an undervoltage/throttle
bitmask plus core voltage and SoC temperature via `vcgencmd`. A rising
undervoltage flag is the best early warning that a battery is sagging.

The recorder samples this every `--status-every` seconds **and** at every event,
writing to `status_log.csv`:

| Column | Meaning |
|--------|---------|
| `time`, `reason` | timestamp; `heartbeat`, `event:<id>`, or `shutdown` |
| `fps_measured` | real FPS over the last window |
| `ring_frames` | frames currently in the RAM ring |
| `events_total` | events saved so far |
| `throttled_hex` | raw `get_throttled` bitmask |
| `warnings` | decoded flags (e.g. `Under-voltage detected`) |
| `core_volt_v` | core rail voltage |
| `soc_temp_c` | SoC temperature |
| `usb_free_bytes` / `usb_free_h` | remaining space on the output drive |
| `sensor_kind` / `sensor_distance_km` / `sensor_energy` | sensor data at trigger |

Throttle-mask bits: 0 undervoltage now, 1 ARM freq capped, 2 throttled now,
3 soft-temp limit; bits 16–19 are the same "since boot" flags.

Quick manual check:
```bash
vcgencmd get_throttled     # 0x0 = healthy
vcgencmd measure_volts core
vcgencmd measure_temp
```
For a true battery gauge, add an INA219/INA260 on the battery rail — the
snapshot structure makes it an easy addition.

---

## 11. Run at boot

`/etc/systemd/system/lightning.service`:
```ini
[Unit]
Description=Lightning camera recorder
After=multi-user.target

[Service]
ExecStart=/home/pi/lightning_cam/venv/bin/python /home/pi/lightning_cam/lightning_run.py --output /media/pi/USB/lightning_events --outdoor
WorkingDirectory=/home/pi/lightning_cam
Restart=on-failure
User=pi

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable --now lightning.service
journalctl -u lightning.service -f
```
Note the `ExecStart` uses the venv's Python directly (no activation needed).

---

## 12. Storage math

At 640×400, 8-bit mono, 120 FPS: ~300 KB/frame decoded → ~108 MB per raw 3 s
event as `.npy`. `mp4` compresses to a few MB; the **raw MJPEG** path stores the
camera's compressed JPEGs (~1/10th the decoded size), which is also what the
ring buffer holds in RAM. Size your USB drive and `--status-every` accordingly.

---

## 13. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `Failed to add edge detection` | Bookworm + RPi.GPIO — install `lgpio` (driver prefers it automatically). |
| Sensor registers all `0x00`/`0xFF`, `NOT RESPONDING` | wrong CS, SI strap, power, or cold solder joint; try `--spi-dev 1`, or I²C + `i2cdetect -y 1`. |
| `i2cdetect` empty on both buses | power/ground problem — verify VCC = 3.3 V, GND continuity, EN pin high. |
| FPS ~50 at 640×400 | CPU decode bound — use the **raw path** (`linuxpy`), or 320×240, or lower res. |
| Constant disturbers, flat calibration, constant `mean_energy` | nearby interferer (Pi/PSU/USB) — relocate the sensor, use battery/linear PSU. |
| Brightness pumps / JPEG size varies >15% | auto-exposure active — set `--exposure`/`--gain` manually. |
| Few frames per strike | frame rate too low — raise FPS via the raw path so 5–15 frames span the glow. |
| No USB auto-detected | mount it (§4) or pass `--output`. |
| `linuxpy` teardown error | use the current `RawMJPEGCamera` (explicit stream close) — not `v4l2py`. |

---

## 14. Future hardware ideas

Once the trigger works and you've caught strikes with this rig, these improve
quality — change one thing at a time.

**Camera-side (rough impact order):**
- **Faster host (Pi 5 / mini-PC):** the USB link ceilings ~122 fps and the Pi
  3B+ can't decode much in real time. A faster host unlocks the camera's full
  240 fps and higher resolutions with live decode.
- **Camera with onboard H.264/H.265 or higher native FPS:** removes the decode
  problem entirely; the OV9281 sensor itself is excellent (global shutter).
- **Wider, faster lens (low f-number):** better odds the channel is in-frame,
  brighter channel, shorter/crisper exposures.
- **Solid mount aimed at the storm quadrant**, using the AS3935 distance to aim.

**System-side:**
- **Separate the sensor from the camera host** (own MCU or long cable, away from
  USB/PSU) — also fixes the interference in §7.
- **Two cameras a known distance apart** → triangulate the 3D path and true
  distance, not just a 2D projection.
- **Photodiode + comparator as a fast optical trigger** → fires on the actual
  flash in microseconds, far faster than RF or frame-based detection, if you
  later want to chase the leader.
