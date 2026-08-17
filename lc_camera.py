"""
lc_camera.py -- OV9281 USB camera capture + RAM ring buffer.

Two layers:
  * CameraV4L2    : opens /dev/videoN with OpenCV (V4L2 backend), pulls frames,
                    reports the negotiated format / FPS.
  * RingBuffer    : fixed-duration circular buffer of (timestamp, frame) kept
                    entirely in RAM, so the pre-trigger window is always
                    available without touching the USB drive.

We use OpenCV's VideoCapture because it is the most reliable way to talk to a
UVC/MJPEG camera on the Pi without pulling in a large framework. For the
global-shutter OV9281 you typically want an uncompressed mono format (GREY /
Y8) or MJPEG; the actual modes MUST be confirmed with:

    v4l2-ctl --list-formats-ext -d /dev/video0
"""

import time
import threading
from collections import deque

try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False

import numpy as np


# Map of fourcc strings we might request. MJPG is broadly supported; GREY/Y8
# gives raw 8-bit mono if the camera exposes it.
FOURCC = {
    "MJPG": "MJPG",
    "YUYV": "YUYV",
    "GREY": "GREY",   # 8-bit monochrome
    "Y8":   "Y8  ",
}


class CameraV4L2:
    def __init__(self, device="/dev/video0", width=640, height=480, fps=120,
                 fourcc="MJPG", mono=True, logger=None):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.mono = mono
        self.log = logger
        self.cap = None
        self.available = _HAVE_CV2

        if not _HAVE_CV2 and logger:
            logger.warning("OpenCV (cv2) not installed -- `pip install opencv-python`")

    # ------------------------------------------------------------------ #
    def open(self) -> bool:
        if not self.available:
            return False

        # device may be "/dev/video0" or an index like 0
        dev = self.device
        if isinstance(dev, str) and dev.startswith("/dev/video"):
            dev_index = int(dev.replace("/dev/video", ""))
        else:
            dev_index = int(dev)

        self.cap = cv2.VideoCapture(dev_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            if self.log:
                self.log.error("Could not open camera %s", self.device)
            return False

        cc = cv2.VideoWriter_fourcc(*FOURCC.get(self.fourcc, "MJPG"))
        self.cap.set(cv2.CAP_PROP_FOURCC, cc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Small internal buffer -> lower latency, fresher frames.
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if self.log:
            self.log.info("Camera opened: requested %dx%d @ %d FPS (%s). "
                          "Negotiated: %s",
                          self.width, self.height, self.fps, self.fourcc,
                          self.describe())
        return True

    def describe(self) -> str:
        if not self.cap:
            return "not open"
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = self.cap.get(cv2.CAP_PROP_FPS)
        raw = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        cc = "".join([chr((raw >> (8 * i)) & 0xFF) for i in range(4)])
        return f"{w}x{h} @ {f:.1f} FPS fourcc={cc.strip()}"

    def read(self):
        """Return (ok, frame). Frame is grayscale if mono=True."""
        if not self.cap:
            return False, None
        ok, frame = self.cap.read()
        if ok and self.mono and frame is not None and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return ok, frame

    def measure_fps(self, seconds=3.0) -> dict:
        """Grab frames for N seconds and report the real achieved FPS."""
        if not self.cap:
            return {"ok": False, "reason": "camera not open"}
        n = 0
        t0 = time.time()
        drops = 0
        while time.time() - t0 < seconds:
            ok, _ = self.read()
            if ok:
                n += 1
            else:
                drops += 1
        dt = time.time() - t0
        return {"ok": True, "frames": n, "seconds": round(dt, 3),
                "fps": round(n / dt, 2) if dt else 0, "read_failures": drops}

    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None


class RingBuffer:
    """
    Time-bounded circular buffer holding the most recent `pre_seconds` of frames.

    Stored as (timestamp, frame_ndarray). Memory use is roughly:
        pre_seconds * fps * frame_bytes
    e.g. 1.5 s * 120 * 307 KB  ~= 55 MB for the pre-roll at 640x480 mono.
    """

    def __init__(self, pre_seconds=1.5, fps=120, logger=None):
        self.pre_seconds = pre_seconds
        self.maxlen = max(1, int(pre_seconds * fps * 1.2))  # +20% headroom
        self.buf = deque(maxlen=self.maxlen)
        self.lock = threading.Lock()
        self.log = logger

    def push(self, ts, frame):
        with self.lock:
            self.buf.append((ts, frame))

    def snapshot_pre(self, trigger_ts):
        """Return frames within pre_seconds before the trigger."""
        cutoff = trigger_ts - self.pre_seconds
        with self.lock:
            return [(t, f) for (t, f) in self.buf if t >= cutoff]

    def __len__(self):
        with self.lock:
            return len(self.buf)
