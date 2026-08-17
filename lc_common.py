"""
lc_common.py -- shared helpers for the lightning-camera project.

Contains:
  * PowerMonitor  : reads Raspberry Pi undervoltage / throttle / temp / core volts
  * find_usb_mounts() : list mounted removable drives
  * setup_logger()  : consistent logging to console + rotating file
  * now_iso() / ts_compact() : timestamp helpers

Nothing here depends on the camera or the sensor, so it can be imported by
every script (tests + final runner) without pulling in heavy deps.
"""

import os
import csv
import glob
import time
import logging
import subprocess
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler


# --------------------------------------------------------------------------- #
# Timestamps
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    """ISO-8601 local timestamp with seconds, e.g. 2026-08-17T14:30:05."""
    return datetime.now().replace(microsecond=0).isoformat()


def ts_compact() -> str:
    """Filesystem-safe timestamp, e.g. 20260817_143005_123 (ms precision)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_") + f"{int(time.time()*1000)%1000:03d}"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logger(name: str, logfile: str | None = None, level=logging.INFO) -> logging.Logger:
    """Return a logger that writes to the console and (optionally) a rotating file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:                      # avoid duplicate handlers on re-import
        return logger

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if logfile:
        os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)
        fh = RotatingFileHandler(logfile, maxBytes=5_000_000, backupCount=5)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# --------------------------------------------------------------------------- #
# Raspberry Pi power / thermal monitoring
# --------------------------------------------------------------------------- #
class PowerMonitor:
    """
    Reads Raspberry Pi power/thermal health via `vcgencmd`.

    The Pi cannot report the *input voltage* from the PSU / battery directly, but
    it exposes:
      * get_throttled   -> a 20-bit flag word (undervoltage, throttling, ...)
      * measure_volts   -> internal core voltage rails
      * measure_temp    -> SoC temperature

    Undervoltage flags are the best available proxy for "battery is sagging".
    So when running on battery, a rising undervoltage/throttle count is your
    early-warning that the pack is getting low or the cabling is marginal.
    """

    # bit -> (short_key, human message)   (per RPi firmware documentation)
    FLAGS = {
        0:  ("uv_now",         "Under-voltage detected"),
        1:  ("freq_cap_now",   "ARM frequency capped"),
        2:  ("throttled_now",  "Currently throttled"),
        3:  ("temp_limit_now", "Soft temperature limit active"),
        16: ("uv_since",       "Under-voltage has occurred since boot"),
        17: ("freq_cap_since", "ARM frequency capping has occurred since boot"),
        18: ("throttled_since","Throttling has occurred since boot"),
        19: ("temp_limit_since","Soft temperature limit has occurred since boot"),
    }

    def __init__(self, logger: logging.Logger | None = None):
        self.log = logger
        self.available = self._which("vcgencmd") is not None
        if not self.available and self.log:
            self.log.warning("vcgencmd not found -- power monitoring disabled "
                             "(are you running off-Pi?).")

    @staticmethod
    def _which(cmd):
        for p in os.environ.get("PATH", "").split(os.pathsep) + ["/usr/bin", "/opt/vc/bin"]:
            f = os.path.join(p, cmd)
            if os.path.isfile(f) and os.access(f, os.X_OK):
                return f
        return None

    def _vcgencmd(self, *args) -> str | None:
        if not self.available:
            return None
        try:
            out = subprocess.check_output(["vcgencmd", *args],
                                          stderr=subprocess.DEVNULL, timeout=5)
            return out.decode(errors="replace").strip()
        except Exception as e:
            if self.log:
                self.log.debug("vcgencmd %s failed: %s", args, e)
            return None

    def get_throttled_raw(self) -> int | None:
        """Return the raw throttled bitmask as an int, or None."""
        out = self._vcgencmd("get_throttled")          # e.g. 'throttled=0x50005'
        if not out or "=" not in out:
            return None
        try:
            return int(out.split("=")[1], 16)
        except ValueError:
            return None

    def decode_flags(self, mask: int) -> dict:
        """Return {short_key: bool} for every known flag bit."""
        return {key: bool(mask & (1 << bit)) for bit, (key, _) in self.FLAGS.items()}

    def active_warnings(self, mask: int) -> list[str]:
        """Human-readable list of currently/historically raised flags."""
        return [msg for bit, (_, msg) in self.FLAGS.items() if mask & (1 << bit)]

    def core_volts(self) -> float | None:
        out = self._vcgencmd("measure_volts", "core")   # e.g. 'volt=0.8563V'
        if out and "=" in out:
            try:
                return float(out.split("=")[1].rstrip("V"))
            except ValueError:
                return None
        return None

    def temp_c(self) -> float | None:
        out = self._vcgencmd("measure_temp")             # e.g. "temp=47.2'C"
        if out and "=" in out:
            try:
                return float(out.split("=")[1].split("'")[0])
            except ValueError:
                return None
        return None

    def snapshot(self) -> dict:
        """
        One consolidated reading. Always returns a dict (fields None if
        unavailable), so callers can log it unconditionally.
        """
        mask = self.get_throttled_raw()
        snap = {
            "time": now_iso(),
            "throttled_hex": None if mask is None else f"0x{mask:X}",
            "core_volt_v": self.core_volts(),
            "soc_temp_c": self.temp_c(),
        }
        if mask is not None:
            snap.update(self.decode_flags(mask))
        return snap


# --------------------------------------------------------------------------- #
# CSV status logger (for the periodic "what is going on" record)
# --------------------------------------------------------------------------- #
class StatusCsv:
    """Append-only CSV writer that lazily writes a header on first row."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._header_written = os.path.exists(path) and os.path.getsize(path) > 0
        self._fields = None

    def write(self, row: dict):
        new_file = not self._header_written
        with open(self.path, "a", newline="") as f:
            if self._fields is None:
                self._fields = list(row.keys())
            w = csv.DictWriter(f, fieldnames=self._fields, extrasaction="ignore")
            if new_file:
                w.writeheader()
                self._header_written = True
            w.writerow(row)


# --------------------------------------------------------------------------- #
# USB / removable-media discovery
# --------------------------------------------------------------------------- #
def find_usb_mounts() -> list[dict]:
    """
    Return a list of currently mounted removable filesystems, each as:
        {"device": "/dev/sda1", "mount": "/media/pi/USB", "fstype": "vfat",
         "size": <bytes|None>, "free": <bytes|None>}

    Strategy:
      1. Prefer parsing `lsblk` (JSON) for anything under /media, /mnt, /run/media.
      2. Fall back to scanning common mount roots on disk.
    This is best-effort and never raises -- an empty list just means "nothing found".
    """
    mounts = []

    # ---- attempt 1: lsblk --------------------------------------------------
    try:
        import json
        out = subprocess.check_output(
            ["lsblk", "-J", "-o", "NAME,MOUNTPOINT,FSTYPE,RM,SIZE,PATH"],
            stderr=subprocess.DEVNULL, timeout=5).decode()
        data = json.loads(out)

        def walk(dev):
            mp = dev.get("mountpoint")
            rm = str(dev.get("rm", "0")) in ("1", "True", "true")
            if mp and rm and mp not in ("/", "/boot"):
                mounts.append({
                    "device": dev.get("path") or ("/dev/" + dev.get("name", "?")),
                    "mount": mp,
                    "fstype": dev.get("fstype"),
                    "size": _du(mp)[0],
                    "free": _du(mp)[1],
                })
            for child in dev.get("children", []) or []:
                walk(child)

        for d in data.get("blockdevices", []):
            walk(d)
    except Exception:
        pass

    # ---- attempt 2: scan mount roots --------------------------------------
    if not mounts:
        roots = ["/media", "/mnt", "/run/media"]
        seen = set()
        for root in roots:
            for path in glob.glob(os.path.join(root, "**"), recursive=True):
                if os.path.ismount(path) and path not in seen:
                    seen.add(path)
                    size, free = _du(path)
                    mounts.append({"device": "?", "mount": path,
                                   "fstype": None, "size": size, "free": free})

    return mounts


def _du(path):
    """Return (total_bytes, free_bytes) for a path, or (None, None)."""
    try:
        st = os.statvfs(path)
        return st.f_frsize * st.f_blocks, st.f_frsize * st.f_bavail
    except Exception:
        return None, None


def human_bytes(n) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def resolve_output_dir(cli_value: str | None, logger=None) -> str:
    """
    Decide where events/logs go.
      * If cli_value given -> use it (create if needed).
      * Else, if a USB mount is found -> use its first mount + /lightning_events.
      * Else -> ./lightning_events in the CWD.
    """
    if cli_value:
        os.makedirs(cli_value, exist_ok=True)
        return cli_value

    usb = find_usb_mounts()
    if usb:
        target = os.path.join(usb[0]["mount"], "lightning_events")
        os.makedirs(target, exist_ok=True)
        if logger:
            logger.info("Auto-selected USB output: %s (free %s)",
                        target, human_bytes(usb[0]["free"]))
        return target

    target = os.path.join(os.getcwd(), "lightning_events")
    os.makedirs(target, exist_ok=True)
    if logger:
        logger.warning("No USB found -- writing to local dir %s", target)
    return target
