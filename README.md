# Lightning Camera — OV9281 + AS3935 on Raspberry Pi 3B+

A trigger-based recorder: an **AS3935 RF lightning sensor** fires an interrupt,
and an **OV9281 global-shutter USB camera** saves a short clip spanning the
moment of the strike (pre-roll from a RAM ring buffer + post-roll). The live
stream is **never** continuously written to disk — only triggered events land
on the USB drive.

```
   Lightning → AS3935 (IRQ) → Raspberry Pi 3B+
                                   │
         camera frames ──► RAM ring buffer (last PRE seconds)
                                   │  on trigger
                          keep capturing POST seconds
                                   │
                        [pre-roll | post-roll] = one event
                                   │
                             External USB drive
```

---

## 1. Files

| File | Purpose |
|------|---------|
| `lc_common.py` | Shared helpers: power monitor, USB discovery, logging, CSV status. |
| `lc_sensor.py` | `AS3935Sensor` driver (SPI/I²C, IRQ callback). |
| `lc_camera.py` | `CameraV4L2` capture + `RingBuffer` (RAM pre-roll). |
| `tests/test_sensor.py` | Verify the sensor and print live events. |
| `tests/test_camera.py` | Verify the camera and measure real FPS. |
| `tests/pressure_test.py` | Stress camera + sensor + power together and report health. |
| `lightning_run.py` | The always-on recorder (ring buffer + triggered saves + heartbeat log). |

---

## 2. Install (on the Raspberry Pi)

Recent Raspberry Pi OS (Bookworm) is **externally managed** — system-wide
`pip3 install` is blocked (`error: externally-managed-environment`). Use a
virtual environment. This is the recommended path:

```bash
cd ~/lightning_cam

# system packages + V4L tools
sudo apt update
sudo apt install -y python3-full python3-venv v4l-utils

# venv that can still fall back to system-provided modules (RPi.GPIO etc.)
python3 -m venv --system-site-packages venv
source venv/bin/activate

pip install --upgrade pip
pip install opencv-python numpy spidev lgpio smbus2
```

**GPIO on Bookworm:** the sensor uses **`lgpio`**, which talks to the modern
`gpiochip` character device. The older `RPi.GPIO` library relies on the sysfs
interface that Bookworm removed, so its edge detection fails with
`RuntimeError: Failed to add edge detection`. `lgpio` is the correct choice on
current Pi OS; the driver falls back to `RPi.GPIO` automatically on older
systems if `lgpio` isn't present. If apt provides it, `sudo apt install -y
python3-lgpio` also works.

The `--system-site-packages` flag matters on the Pi: some libraries (parts of
the camera stack, sometimes `lgpio`) already ship via apt, and this lets the
venv use them if a pip wheel is troublesome.

> **`opencv-python` slow or failing to build on a Pi 3B+?** Use the
> precompiled apt build instead — it's ARM-native and generally faster:
> ```bash
> deactivate
> sudo apt install -y python3-opencv
> cd ~/lightning_cam && rm -rf venv
> python3 -m venv --system-site-packages venv   # inherits system cv2
> source venv/bin/activate
> pip install numpy spidev RPi.GPIO smbus2       # cv2 comes from system
> python -c "import cv2; print('cv2', cv2.__version__)"
> ```

**Every session, activate the venv before running anything:**

```bash
cd ~/lightning_cam
source venv/bin/activate
python tests/test_camera.py --list
```

*(If you deliberately want a system-wide install instead of a venv, you can
append `--break-system-packages` to `pip3 install`, at the documented risk of
disturbing OS Python packages. The venv above is safer.)*

### Enable the buses

```bash
sudo raspi-config    # Interface Options → enable SPI (and/or I2C)
# verify (SPI CE0/CE1 and the I2C buses should appear):
ls /dev/spidev*      # e.g. /dev/spidev0.0  /dev/spidev0.1
ls /dev/i2c*         # e.g. /dev/i2c-1
```

`spidev0.0` is chip-select **CE0** (`--spi-dev 0`), `spidev0.1` is **CE1**
(`--spi-dev 1`) — make sure this matches which CE line your sensor CS is wired
to.

---

## 3. Wiring (SPI, as planned)

| AS3935 module | Raspberry Pi 3B+ | BCM |
|---------------|------------------|-----|
| VCC  | 3.3 V | — |
| GND  | GND | — |
| SCLK | SPI0_SCLK | GPIO 11 |
| MOSI | SPI0_MOSI | GPIO 10 |
| MISO | SPI0_MISO | GPIO 9 |
| CS   | SPI0_CE0 | GPIO 8 |
| IRQ  | (example) | GPIO 17 |

> ⚠️ **Verify the physical GY-AS3935 pinout before wiring.** Breakout boards
> relabel pins and some ship in I²C mode by default. The chip supports both
> SPI and I²C; this driver assumes the SPI wiring above. If the register dump
> in `test_sensor.py` reads all `0x00` or all `0xFF`, the bus/CS is wrong.

---

## 4. Find your USB drive

List block devices and mount points:

```bash
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT,RM,LABEL
```
`RM=1` marks removable media. Other useful commands:

```bash
# What is mounted where, removable filesystems only:
findmnt -t vfat,exfat,ext4 -o SOURCE,TARGET,FSTYPE,SIZE

# Kernel view of what was just plugged in:
dmesg | tail -n 20

# By-id / by-label symlinks (stable across reboots):
ls -l /dev/disk/by-id/ /dev/disk/by-label/

# Free space on a mount:
df -h /media/pi/*
```

Mount one manually if it isn't auto-mounted:

```bash
sudo mkdir -p /media/pi/USB
sudo mount /dev/sda1 /media/pi/USB          # use your device from lsblk
# exFAT needs:  sudo apt install exfat-fuse exfatprogs
```

The scripts can **auto-detect** a removable mount — just omit `--output` and
the recorder picks the first USB it finds (falling back to `./lightning_events`).
You can list what it sees with:

```bash
python3 -c "from lc_common import find_usb_mounts,human_bytes; \
[print(u['device'],u['mount'],human_bytes(u['free'])) for u in find_usb_mounts()]"
```

---

## 5. Testing workflow

**Step 1 — camera modes and FPS.** Always check real V4L2 modes first:

```bash
v4l2-ctl --list-formats-ext -d /dev/video0
python3 tests/test_camera.py --list                       # same, via the script
python3 tests/test_camera.py --width 640 --height 480 --fps 120 --fourcc MJPG
python3 tests/test_camera.py --fourcc GREY --save sample.png   # raw mono + snapshot
```
The script reports the **negotiated** format (what the driver actually gave)
and the **measured** FPS, which is often below the advertised 120.

**Step 2 — sensor.** Confirm the bus and watch live events:

```bash
python3 tests/test_sensor.py                              # SPI0/CE0, IRQ GPIO17
python3 tests/test_sensor.py --outdoor --noise-floor 2 --seconds 120
python3 tests/test_sensor.py --bus i2c --irq-gpio 17
```
A piezo lighter clicked near the antenna usually produces a **disturber**
interrupt — handy to prove the IRQ path works without a real storm.

**Step 3 — pressure test (debugging).** Run everything at once and read the
summary (sustained FPS, jitter, dropped reads, sensor hits, and every
undervoltage/throttle flag seen):

```bash
python3 tests/pressure_test.py --seconds 120
python3 tests/pressure_test.py --seconds 300 --fps 120 --fourcc GREY
python3 tests/pressure_test.py --no-sensor --seconds 180   # camera+power only
```
Exit code is non-zero if the Pi throttled or FPS fell far below target — good
for scripting a pass/fail.

---

## 6. Running the recorder

```bash
# Auto-pick USB for output, defaults: 1.5 s pre + 1.5 s post, 640x480@120, mp4
python3 lightning_run.py

# Explicit USB path, raw frames, outdoor sensor preset, status row every 60 s
python3 lightning_run.py \
    --output /media/pi/USB/lightning_events \
    --format npy --outdoor --status-every 60

# Bench test with no storm: run camera-only and fire manual triggers
python3 lightning_run.py --no-sensor &
kill -USR1 $!        # forces one event to be recorded
```

### Key arguments
Run `python3 lightning_run.py --help` for the full list.

| Arg | Default | Meaning |
|-----|---------|---------|
| `--pre` / `--post` | 1.5 / 1.5 | seconds before/after the trigger |
| `--fps` | 120 | target capture rate |
| `--width/--height` | 640/480 | resolution |
| `--fourcc` | MJPG | MJPG / YUYV / GREY / Y8 |
| `--format` | mp4 | `mp4` (small), `npy` (raw lossless), `png` (per-frame) |
| `--output` | auto | output dir; auto-picks USB if omitted |
| `--status-every` | 120 | seconds between heartbeat status rows |
| `--no-sensor` | off | run without AS3935 (manual/`SIGUSR1` trigger) |
| `--outdoor` | off | less-sensitive AFE preset for rooftop/field |
| `--trigger-on-disturber` | off | also save on disturber IRQs (debug) |

### What lands on the drive
Per event, under `<output>/events/`:
* `event_<timestamp>.mp4` (or `.npy` / `_frames/` PNG folder)
* `event_<timestamp>.json` — sidecar with the sensor reading (kind, distance,
  energy) **and the power/throttle state at the trigger moment**.

Plus, at the output root:
* `status_log.csv` — the periodic "what's going on" record (see below).
* `run.log` — full rotating text log.

---

## 7. Power / battery monitoring

**Can we record the Pi's power status over time on battery? Yes.**

The Pi cannot report the raw input voltage from a battery/PSU directly, but its
firmware exposes an **undervoltage / throttling** bitmask plus core voltage and
SoC temperature via `vcgencmd`. When running on battery, a rising undervoltage
or throttle flag is your early warning that the pack is sagging or the cabling
is marginal — the most useful signal you can get without extra hardware.

The recorder samples this every `--status-every` seconds **and** at every saved
event, writing to `status_log.csv`:

| Column | Meaning |
|--------|---------|
| `time`, `reason` | timestamp; `heartbeat`, `event:<id>`, or `shutdown` |
| `fps_measured` | real FPS over the last window |
| `ring_frames` | frames currently in the RAM ring |
| `events_total` | events saved so far |
| `throttled_hex` | raw `get_throttled` bitmask |
| `warnings` | decoded flags (e.g. `Under-voltage detected`) |
| `core_volt_v` | core rail voltage (`vcgencmd measure_volts`) |
| `soc_temp_c` | SoC temperature |
| `usb_free_bytes` / `usb_free_h` | remaining space on the output drive |
| `sensor_kind` / `sensor_distance_km` / `sensor_energy` | sensor data at trigger |

Bit meanings in the throttle mask: bit 0 undervoltage now, bit 1 ARM freq
capped, bit 2 throttled now, bit 3 soft-temp limit; bits 16–19 are the same
"has occurred since boot" flags. For a truly accurate battery gauge, add an
INA219/INA260 current+voltage sensor on the battery rail — the driver structure
here makes it easy to log alongside the existing snapshot.

Quick manual check any time:

```bash
vcgencmd get_throttled     # 0x0 = healthy
vcgencmd measure_volts core
vcgencmd measure_temp
```

---

## 8. Run at boot (optional)

`/etc/systemd/system/lightning.service`:

```ini
[Unit]
Description=Lightning camera recorder
After=multi-user.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/lightning_cam/lightning_run.py --output /media/pi/USB/lightning_events --outdoor
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

---

## 9. Storage math (from the plan)

At 640×480, 8-bit mono, 120 FPS: ~307 KB/frame → 360 frames for 3 s →
**~111 MB per raw event**. `mp4` compresses this to a few MB; `npy`/`png` keep
it lossless and large. Size your USB drive and `--status-every` accordingly.

---

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| Sensor register dump all `0x00`/`0xFF` | wrong CS pin, wrong bus, or board in I²C mode — recheck wiring/`--bus`. |
| FPS far below 120 | use MJPG, lower resolution, a powered USB port; verify with `v4l2-ctl`. |
| Frequent `Under-voltage detected` | weak PSU/battery or thin cable; use a solid 5 V/2.5 A+ supply. |
| No USB auto-detected | mount it (`§4`) or pass `--output`. |
| Constant disturber IRQs | raise `--noise-floor`, use `--outdoor`, move away from motors/switching supplies. |
| No events during a storm | AS3935 range ~40 km and needs a clear-ish RF environment; test the IRQ path with a piezo lighter first. |
