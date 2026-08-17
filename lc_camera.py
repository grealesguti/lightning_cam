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

    @staticmethod
    def decode_jpeg(jpeg_bytes, mono=True):
        """Decode JPEG bytes (from the raw V4L2 capturer) into an image array."""
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        flag = cv2.IMREAD_GRAYSCALE if mono else cv2.IMREAD_COLOR
        return cv2.imdecode(arr, flag)

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


class RawMJPEGCamera:
    """
    Capture RAW (undecoded) MJPEG frames straight from V4L2, bypassing OpenCV's
    per-frame JPEG decode.

    Why: on a Pi 3B+, OpenCV's cap.read() decodes every frame on the CPU, which
    caps you at ~50 fps at 640x400 even though the USB link delivers 120+ fps.
    Storing the compressed JPEG bytes instead skips decode entirely -- measured
    ~238 fps at 640x400 on a Pi 3B+ vs ~50 fps decoded. Decode only the frames
    you keep, at save/review time, with CameraV4L2.decode_jpeg().

    Backend: `linuxpy.video` (the maintained successor to v4l2py). We drive the
    streaming lifecycle explicitly via a VideoCapture context so start/stop is
    clean and there is no teardown exception.

    If linuxpy isn't installed, `.available` is False -- fall back to CameraV4L2.
    """

    def __init__(self, device="/dev/video0", width=640, height=400, fps=120,
                 exposure=None, gain=None, auto_exposure=False, logger=None):
        """
        exposure : manual exposure time in device units (v4l2
                   exposure_time_absolute, typically 100us steps). None leaves
                   it alone. For lightning use MANUAL exposure so brightness is
                   stable frame-to-frame (auto-exposure pumps and ruins motion
                   analysis).
        gain     : manual analog gain. None leaves it alone.
        auto_exposure : if False (default), disable auto-exposure so your
                   manual exposure/gain stick.
        """
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.exposure = exposure
        self.gain = gain
        self.auto_exposure = auto_exposure
        self.log = logger
        self._dev = None
        self._cap = None          # VideoCapture context
        self._frames = None       # frame iterator
        try:
            import linuxpy.video.device as _lv
            self._lv = _lv
            self.available = True
        except Exception:
            self._lv = None
            self.available = False
            if logger:
                logger.warning("linuxpy not installed -- raw MJPEG capture "
                               "unavailable. `pip install linuxpy` to enable "
                               "the fast path; falling back to CameraV4L2.")

    def _dev_id(self):
        d = self.device
        if isinstance(d, str) and d.startswith("/dev/video"):
            return int(d.replace("/dev/video", ""))
        return int(d)

    def _apply_controls(self):
        """
        Set manual exposure/gain via v4l2 controls. Uses v4l2-ctl as the
        portable path (control names vary by driver). Best-effort: logs what it
        set and never raises.
        """
        import subprocess
        cmds = []
        # Turn OFF auto-exposure first (value 1 = manual mode in V4L2's
        # exposure_auto menu; drivers differ, so we try both common names).
        if not self.auto_exposure:
            cmds.append(("exposure_auto", 1))
            cmds.append(("auto_exposure", 1))
        if self.exposure is not None:
            cmds.append(("exposure_time_absolute", self.exposure))
            cmds.append(("exposure_absolute", self.exposure))
        if self.gain is not None:
            cmds.append(("gain", self.gain))
        for name, val in cmds:
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", self.device if isinstance(self.device, str)
                     else f"/dev/video{self.device}",
                     "--set-ctrl", f"{name}={val}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=3)
            except Exception:
                pass
        if self.log and (self.exposure is not None or self.gain is not None):
            self.log.info("Applied manual controls: exposure=%s gain=%s "
                          "auto_exposure=%s", self.exposure, self.gain,
                          self.auto_exposure)

    def open(self) -> bool:
        if not self.available:
            return False
        lv = self._lv
        self._dev = lv.Device.from_id(self._dev_id())
        self._dev.open()
        # request MJPEG at the desired geometry
        self._dev.set_format(lv.BufferType.VIDEO_CAPTURE,
                             self.width, self.height, lv.PixelFormat.MJPEG)
        try:
            self._dev.set_fps(lv.BufferType.VIDEO_CAPTURE, self.fps)
        except Exception:
            pass
        # apply manual exposure/gain BEFORE streaming so the first frames obey
        self._apply_controls()
        # start streaming via a VideoCapture context we hold open
        self._cap = lv.VideoCapture(self._dev)
        self._cap.open()                     # begins streaming
        self._frames = iter(self._cap)
        if self.log:
            self.log.info("RawMJPEGCamera open: %dx%d MJPG @%d (raw, no decode).",
                          self.width, self.height, self.fps)
        return True

    def read_raw(self):
        """Return (ok, jpeg_bytes) for one frame, without decoding."""
        if self._frames is None:
            return False, None
        try:
            frame = next(self._frames)
            return True, bytes(frame.data)
        except StopIteration:
            return False, None
        except Exception as e:
            if self.log:
                self.log.debug("raw read error: %s", e)
            return False, None

    def measure_fps(self, seconds=3.0) -> dict:
        if self._frames is None:
            return {"ok": False, "reason": "camera not open"}
        n, drops = 0, 0
        t0 = time.time()
        while time.time() - t0 < seconds:
            ok, _ = self.read_raw()
            if ok:
                n += 1
            else:
                drops += 1
        dt = time.time() - t0
        return {"ok": True, "frames": n, "seconds": round(dt, 3),
                "fps": round(n / dt, 2) if dt else 0, "read_failures": drops}

    def close(self):
        # order matters: stop the stream, then close the device
        try:
            if self._cap is not None:
                self._cap.close()
        except Exception:
            pass
        try:
            if self._dev is not None:
                self._dev.close()
        except Exception:
            pass
        self._cap = None
        self._frames = None
        self._dev = None


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
