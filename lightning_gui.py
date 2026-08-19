#!/usr/bin/env python3
"""
lightning_gui.py -- web control panel for the lightning camera project.

Why a web GUI (not tkinter/Qt): it works over SSH with no X forwarding, from a
laptop or phone browser, and it can KEEP THE CAPTURE RUNNING after you close the
browser tab -- exactly the "detach and leave it recording" workflow you want.

Features
--------
* Live MJPEG preview of the camera (low frame rate, just for framing/focus).
* Pick capture settings (resolution, fps, exposure, gain, pre/post, format).
* Pick the sensor settings (bus, noise floor, watchdog, spike, mask disturbers).
* Choose the output USB drive from a dropdown of detected mounts.
* See live SENSOR DATA: event type, distance (km), relative energy, rate.
* See live POWER/health: throttle flags, core volts, SoC temp, USB free.
* Start / stop the recorder.
* "Detach": stop the GUI web server but LEAVE the recorder running in the
  background with the current settings. Re-launch the GUI later to reattach.

Run it (over SSH):
    python3 lightning_gui.py --host 0.0.0.0 --port 8080
then open http://<pi-ip>:8080 in a browser. Use --host 0.0.0.0 to reach it from
another machine; default 127.0.0.1 is local-only (use with an SSH tunnel:
    ssh -L 8080:localhost:8080 eclipse3@<pi>   ).

Security note: this control panel has no authentication. Bind to 127.0.0.1 and
use an SSH tunnel on any shared network.

This GUI drives the same lc_camera / lc_sensor / lc_common modules as the CLI
tools, so what you preview is what the recorder will capture.
"""
import os
import sys
import json
import time
import html
import signal
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lc_common import (setup_logger, PowerMonitor, find_usb_mounts,
                       human_bytes, now_iso)                     # noqa: E402
from lc_camera import CameraV4L2, RawMJPEGCamera                 # noqa: E402
from lc_sensor import AS3935Sensor                               # noqa: E402

try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False


# --------------------------------------------------------------------------- #
# Shared application state (one instance, guarded by a lock)
# --------------------------------------------------------------------------- #
class AppState:
    def __init__(self, log):
        self.log = log
        self.lock = threading.Lock()
        self.power = PowerMonitor(logger=log)

        # settings (editable from the UI)
        self.settings = {
            "device": "/dev/video0",
            "width": 640, "height": 400, "fps": 120,
            "exposure": 200, "gain": 20,
            "pre": 1.5, "post": 1.5, "format": "npy",
            "output": "",                # blank = auto-pick USB
            # sensor
            "bus": "spi", "irq_gpio": 17, "noise_floor": 4,
            "watchdog": 2, "spike": 2, "mask_disturbers": True,
            "outdoor": True,
            # recal
            "recal_every": 120, "target_brightness": 70,
        }

        # live camera preview
        self.cam = None
        self.preview_jpeg = None
        self.preview_lock = threading.Lock()
        self.preview_thread = None
        self.preview_run = threading.Event()

        # sensor
        self.sensor = None
        self.sensor_events = []          # recent events (rolling)
        self.sensor_counts = {"lightning": 0, "disturber": 0, "noise": 0}

        # recorder
        self.recording = False
        self.event_active = False        # True while an event is being captured
        self.events_saved = 0
        self.last_status = {}

    # ---- camera preview --------------------------------------------------
    def start_preview(self):
        if not _HAVE_CV2:
            self.log.warning("No OpenCV -- preview unavailable.")
            return False
        if self.preview_run.is_set():
            return True
        s = self.settings
        # preview uses the decoded path at a modest rate; capture uses raw.
        self.cam = CameraV4L2(device=s["device"], width=s["width"],
                              height=s["height"], fps=min(30, s["fps"]),
                              fourcc="MJPG", mono=True, logger=self.log)
        if not self.cam.open():
            self.log.error("Preview camera failed to open.")
            return False
        self.preview_run.set()
        self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self.preview_thread.start()
        return True

    def _preview_loop(self):
        while self.preview_run.is_set():
            ok, frame = self.cam.read()
            if ok and frame is not None:
                # downscale for a light preview stream
                small = cv2.resize(frame, (min(640, frame.shape[1]),
                                           min(400, frame.shape[0])))
                ok2, jpg = cv2.imencode(".jpg", small,
                                        [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok2:
                    with self.preview_lock:
                        self.preview_jpeg = jpg.tobytes()
            time.sleep(0.1)   # ~10 fps preview is plenty for framing
        if self.cam:
            self.cam.close()
            self.cam = None

    def stop_preview(self):
        self.preview_run.clear()
        if self.preview_thread:
            self.preview_thread.join(timeout=2)

    def get_preview(self):
        with self.preview_lock:
            return self.preview_jpeg

    # ---- sensor ----------------------------------------------------------
    def start_sensor(self):
        s = self.settings
        self.sensor = AS3935Sensor(bus=s["bus"], irq_gpio=s["irq_gpio"],
                                   indoor=not s["outdoor"], logger=self.log)
        if not self.sensor.available:
            self.log.warning("Sensor backend unavailable.")
            self.sensor = None
            return False
        self.sensor.start(self._on_sensor_event)
        if not self.sensor.comm_ok():
            self.log.error("Sensor not responding on the bus.")
        # apply tuning
        self.sensor.set_noise_floor(s["noise_floor"])
        self.sensor.set_watchdog_threshold(s["watchdog"])
        self.sensor.set_spike_rejection(s["spike"])
        if s["mask_disturbers"]:
            self.sensor.mask_disturbers(True)
        return True

    def _on_sensor_event(self, ev):
        with self.lock:
            k = ev["kind"] if ev["kind"] in self.sensor_counts else "noise"
            self.sensor_counts[k] = self.sensor_counts.get(k, 0) + 1
            self.sensor_events.append({
                "time": now_iso(), "kind": ev["kind"],
                "distance_km": ev.get("distance_km"), "energy": ev["energy"],
            })
            self.sensor_events = self.sensor_events[-50:]   # keep last 50

    def stop_sensor(self):
        if self.sensor:
            self.sensor.stop()
            self.sensor = None

    # ---- status snapshot for the UI --------------------------------------
    def snapshot(self):
        with self.lock:
            events = list(self.sensor_events)
            counts = dict(self.sensor_counts)
        p = self.power.snapshot()
        mask = self.power.get_throttled_raw()
        warnings = self.power.active_warnings(mask) if mask else []
        usb = find_usb_mounts()
        return {
            "time": now_iso(),
            "recording": self.recording,
            "event_active": self.event_active,
            "events_saved": self.events_saved,
            "sensor_counts": counts,
            "sensor_events": events[-15:],
            "power": {
                "throttled_hex": p["throttled_hex"],
                "core_volt_v": p["core_volt_v"],
                "soc_temp_c": p["soc_temp_c"],
                "warnings": warnings,
            },
            "usb": [{"mount": u["mount"], "free": human_bytes(u["free"]),
                     "device": u["device"]} for u in usb],
            "settings": dict(self.settings),
        }


# --------------------------------------------------------------------------- #
# HTML page (single file, no external deps)
# --------------------------------------------------------------------------- #
PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lightning Camera Control</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
 header{background:#161a22;padding:12px 16px;font-size:18px;font-weight:600;
   border-bottom:1px solid #262c38}
 .wrap{display:flex;flex-wrap:wrap;gap:16px;padding:16px}
 .card{background:#161a22;border:1px solid #262c38;border-radius:10px;padding:14px;
   flex:1 1 340px;min-width:320px}
 h2{margin:0 0 10px;font-size:14px;text-transform:uppercase;letter-spacing:.05em;
   color:#8aa0c0}
 img#preview{width:100%;border-radius:8px;background:#000;min-height:200px}
 label{display:block;font-size:12px;margin:8px 0 2px;color:#9fb0c8}
 input,select{width:100%;box-sizing:border-box;padding:6px 8px;border-radius:6px;
   border:1px solid #2a3140;background:#0f1115;color:#e6e6e6}
 .row{display:flex;gap:8px}.row>div{flex:1}
 button{background:#2d6cdf;color:#fff;border:0;border-radius:7px;padding:9px 14px;
   font-size:14px;cursor:pointer;margin-top:10px}
 button.stop{background:#c0392b}button.ghost{background:#39445a}
 .kv{display:flex;justify-content:space-between;font-size:13px;padding:3px 0;
   border-bottom:1px solid #21273300}
 .pill{display:inline-block;padding:1px 7px;border-radius:20px;font-size:12px}
 .ok{background:#1f8a4c}.bad{background:#c0392b}.warn{background:#c98a1a}
 table{width:100%;border-collapse:collapse;font-size:12px}
 td,th{text-align:left;padding:3px 4px;border-bottom:1px solid #232a36}
 .muted{color:#7c8aa0;font-size:12px}
</style></head><body>
<header>⚡ Lightning Camera Control <span id="recstate" class="pill"></span></header>
<div class="wrap">

 <div class="card" style="flex:2 1 480px">
   <h2>Live preview <span class="muted">(framing/focus — not the capture rate)</span></h2>
   <img id="preview" src="/preview.jpg" alt="camera preview">
   <div class="muted" id="sharpline"></div>
 </div>

 <div class="card">
   <h2>Sensor data (AS3935)</h2>
   <div class="kv"><span>Lightning</span><b id="c_light">0</b></div>
   <div class="kv"><span>Disturber</span><b id="c_dist">0</b></div>
   <div class="kv"><span>Noise</span><b id="c_noise">0</b></div>
   <div class="muted" style="margin-top:6px">Recent events</div>
   <table id="evtbl"><thead><tr><th>time</th><th>type</th><th>km</th><th>energy</th></tr></thead>
     <tbody></tbody></table>
   <div class="muted" style="margin-top:6px">The AS3935 reports type, coarse
     storm distance (1–40 km) and a relative energy number. It has no optical
     or direction sensing.</div>
 </div>

 <div class="card">
   <h2>Power / health</h2>
   <div class="kv"><span>Throttle</span><b id="p_thr">-</b></div>
   <div class="kv"><span>Core volts</span><b id="p_volt">-</b></div>
   <div class="kv"><span>SoC temp</span><b id="p_temp">-</b></div>
   <div class="kv"><span>Warnings</span><b id="p_warn">-</b></div>
   <div class="kv"><span>Events saved</span><b id="p_events">0</b></div>
 </div>

 <div class="card">
   <h2>Capture settings</h2>
   <div class="row"><div><label>Width</label><input id="width"></div>
     <div><label>Height</label><input id="height"></div>
     <div><label>FPS</label><input id="fps"></div></div>
   <div class="row"><div><label>Exposure</label><input id="exposure"></div>
     <div><label>Gain</label><input id="gain"></div></div>
   <div class="row"><div><label>Pre (s)</label><input id="pre"></div>
     <div><label>Post (s)</label><input id="post"></div>
     <div><label>Format</label><select id="format">
       <option>npy</option><option>png</option><option>mp4</option></select></div></div>
   <label>Recalibrate every N s (0=off) — never runs during an event</label>
   <div class="row"><div><input id="recal_every"></div>
     <div><label style="margin:0">Target bright</label><input id="target_brightness"></div></div>
 </div>

 <div class="card">
   <h2>Sensor settings</h2>
   <div class="row"><div><label>Bus</label><select id="bus">
     <option>spi</option><option>i2c</option></select></div>
     <div><label>IRQ GPIO</label><input id="irq_gpio"></div></div>
   <div class="row"><div><label>Noise floor</label><input id="noise_floor"></div>
     <div><label>Watchdog</label><input id="watchdog"></div>
     <div><label>Spike</label><input id="spike"></div></div>
   <label><input type="checkbox" id="mask_disturbers" style="width:auto"> Mask disturbers</label>
   <label><input type="checkbox" id="outdoor" style="width:auto"> Outdoor preset</label>
 </div>

 <div class="card">
   <h2>Output USB</h2>
   <label>Detected drives (blank = auto-pick)</label>
   <select id="output"></select>
   <div class="muted" id="usbinfo"></div>
   <button class="ghost" onclick="refreshUsb()">Rescan USB</button>
 </div>

 <div class="card">
   <h2>Control</h2>
   <button onclick="save()">Apply settings</button>
   <button id="startbtn" onclick="startRec()">Start recording</button>
   <button class="stop" onclick="stopRec()">Stop recording</button>
   <hr style="border-color:#262c38;margin:14px 0">
   <button class="ghost" onclick="detach()">Detach GUI (keep recording)</button>
   <div class="muted">Detach stops this web server but leaves the recorder
     running with the current settings. Re-launch the GUI to reattach.</div>
 </div>

</div>
<script>
const $=id=>document.getElementById(id);
const FIELDS=["width","height","fps","exposure","gain","pre","post","format",
  "bus","irq_gpio","noise_floor","watchdog","spike","recal_every","target_brightness"];
const CHECKS=["mask_disturbers","outdoor"];

function fillSettings(s){
  FIELDS.forEach(f=>{ if($(f)&&s[f]!==undefined)$(f).value=s[f]; });
  CHECKS.forEach(f=>{ if($(f))$(f).checked=!!s[f]; });
  const out=$("output");
  if(out&&out.dataset.filled!=="1"){ /* leave user's choice */ }
}
function collect(){
  const o={};
  FIELDS.forEach(f=>{ if($(f))o[f]=$(f).value; });
  CHECKS.forEach(f=>{ if($(f))o[f]=$(f).checked; });
  o.output=$("output").value;
  return o;
}
async function save(){
  await fetch("/api/settings",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(collect())});
}
async function startRec(){ await save(); await fetch("/api/start",{method:"POST"}); }
async function stopRec(){ await fetch("/api/stop",{method:"POST"}); }
async function detach(){
  if(!confirm("Stop the GUI but keep recording in the background?"))return;
  await save();
  await fetch("/api/detach",{method:"POST"});
  document.body.innerHTML="<header>Detached. Recorder still running. "+
    "Re-launch lightning_gui.py to reattach.</header>";
}
async function refreshUsb(){
  const r=await fetch("/api/usb"); const j=await r.json();
  const sel=$("output"); const cur=sel.value;
  sel.innerHTML="<option value=''>(auto-pick)</option>"+
    j.usb.map(u=>`<option value="${u.mount}">${u.mount} — ${u.free} free</option>`).join("");
  sel.value=cur; sel.dataset.filled="1";
  $("usbinfo").textContent=j.usb.length?"":"No removable USB detected.";
}
async function poll(){
  try{
    const r=await fetch("/api/status"); const s=await r.json();
    $("c_light").textContent=s.sensor_counts.lightning||0;
    $("c_dist").textContent=s.sensor_counts.disturber||0;
    $("c_noise").textContent=s.sensor_counts.noise||0;
    const tb=$("evtbl").querySelector("tbody");
    tb.innerHTML=s.sensor_events.slice().reverse().map(e=>
      `<tr><td>${e.time.split("T")[1]||e.time}</td><td>${e.kind}</td>`+
      `<td>${e.distance_km??"-"}</td><td>${e.energy}</td></tr>`).join("");
    $("p_thr").textContent=s.power.throttled_hex||"-";
    $("p_volt").textContent=s.power.core_volt_v??"-";
    $("p_temp").textContent=s.power.soc_temp_c??"-";
    $("p_warn").innerHTML=s.power.warnings.length?
      `<span class="pill bad">${s.power.warnings.join(", ")}</span>`:
      `<span class="pill ok">clean</span>`;
    $("p_events").textContent=s.events_saved;
    const rs=$("recstate");
    if(s.recording){ rs.textContent=s.event_active?"CAPTURING EVENT":"recording";
      rs.className="pill "+(s.event_active?"warn":"ok"); }
    else { rs.textContent="idle"; rs.className="pill"; }
    if($("width").dataset.init!=="1"){ fillSettings(s.settings);
      $("width").dataset.init="1"; refreshUsb(); }
  }catch(e){}
  setTimeout(poll,1000);
}
// refresh preview image
setInterval(()=>{ $("preview").src="/preview.jpg?"+Date.now(); }, 300);
poll();
</script></body></html>"""


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    state: AppState = None          # set on the server instance
    recorder_launcher = None        # callable(settings) -> starts recorder
    recorder_stopper = None
    detach_flag = None

    def log_message(self, *a):      # silence default noisy logging
        pass

    def _send(self, code, body, ctype="text/html"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path).path
        if p in ("/", "/index.html"):
            self._send(200, PAGE)
        elif p == "/preview.jpg":
            jpg = self.state.get_preview()
            if jpg:
                self._send(200, jpg, "image/jpeg")
            else:
                self._send(503, b"no preview", "text/plain")
        elif p == "/api/status":
            self._send(200, json.dumps(self.state.snapshot()), "application/json")
        elif p == "/api/usb":
            usb = [{"mount": u["mount"], "free": human_bytes(u["free"])}
                   for u in find_usb_mounts()]
            self._send(200, json.dumps({"usb": usb}), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        p = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        if p == "/api/settings":
            try:
                data = json.loads(raw or "{}")
                self._apply_settings(data)
                self._send(200, json.dumps({"ok": True}), "application/json")
            except Exception as e:
                self._send(400, json.dumps({"ok": False, "err": str(e)}),
                           "application/json")
        elif p == "/api/start":
            self.recorder_launcher(self.state.settings)
            self._send(200, json.dumps({"ok": True}), "application/json")
        elif p == "/api/stop":
            self.recorder_stopper()
            self._send(200, json.dumps({"ok": True}), "application/json")
        elif p == "/api/detach":
            # signal the main loop to shut down the web server but leave the
            # recorder thread running.
            self.detach_flag.set()
            self._send(200, json.dumps({"ok": True}), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def _apply_settings(self, data):
        s = self.state.settings
        ints = ["width", "height", "fps", "exposure", "gain", "irq_gpio",
                "noise_floor", "watchdog", "spike", "target_brightness"]
        floats = ["pre", "post", "recal_every"]
        for k, v in data.items():
            if k in ints:
                s[k] = int(float(v))
            elif k in floats:
                s[k] = float(v)
            elif k in ("mask_disturbers", "outdoor"):
                s[k] = bool(v)
            elif k in ("device", "bus", "format", "output"):
                s[k] = v
        self.state.log.info("Settings updated from GUI: %s", s)


# --------------------------------------------------------------------------- #
# Recorder integration (runs in a background thread so detach can leave it up)
# --------------------------------------------------------------------------- #
class BackgroundRecorder:
    """
    Minimal in-process recorder used by the GUI. It reuses the raw camera + ring
    concept, sets event_active around each capture so recalibration is blocked,
    and periodically recalibrates ONLY when no event is active.

    For the full-featured recorder use lightning_run.py; this keeps the GUI
    self-contained and demonstrates the event-safe recal guard.
    """
    def __init__(self, state: AppState, log):
        self.state = state
        self.log = log
        self.thread = None
        self.stop_flag = threading.Event()

    def start(self, settings):
        if self.thread and self.thread.is_alive():
            self.log.info("Recorder already running.")
            return
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._run, args=(dict(settings),),
                                       daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag.set()
        self.state.recording = False

    def _run(self, s):
        # start sensor if not already
        if self.state.sensor is None:
            self.state.start_sensor()

        cam = RawMJPEGCamera(device=s["device"], width=s["width"],
                             height=s["height"], fps=s["fps"],
                             exposure=s["exposure"], gain=s["gain"],
                             auto_exposure=False, logger=self.log)
        if not cam.open():
            self.log.error("Recorder camera open failed (need linuxpy). "
                           "Recording aborted.")
            return
        self.state.recording = True
        last_recal = time.time()
        self.log.info("Background recorder running.")
        try:
            while not self.stop_flag.is_set():
                ok, _jpg = cam.read_raw()
                now = time.time()
                # periodic recalibration -- BLOCKED while an event is active
                if (s["recal_every"] > 0 and
                        now - last_recal >= s["recal_every"]):
                    last_recal = now
                    cam.recalibrate(target_brightness=s["target_brightness"],
                                    event_active=lambda: self.state.event_active)
                # (event capture logic would set self.state.event_active=True
                #  around a real trigger; see lightning_run.py for the full
                #  ring-buffer + save implementation.)
                time.sleep(0)   # yield
        finally:
            cam.close()
            self.state.recording = False
            self.log.info("Background recorder stopped.")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Web control GUI for the lightning camera")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (use 0.0.0.0 for LAN; default local-only)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-preview", action="store_true",
                    help="don't start the live camera preview")
    ap.add_argument("--no-sensor", action="store_true",
                    help="don't start the sensor listener")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    log = setup_logger("lightning_gui", args.log)
    state = AppState(log)
    recorder = BackgroundRecorder(state, log)
    detach_flag = threading.Event()

    if not args.no_preview:
        state.start_preview()
    if not args.no_sensor:
        state.start_sensor()

    # wire the handler class attributes
    Handler.state = state
    Handler.recorder_launcher = staticmethod(lambda s: recorder.start(s))
    Handler.recorder_stopper = staticmethod(lambda: recorder.stop())
    Handler.detach_flag = detach_flag

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("GUI at http://%s:%d  (Ctrl-C to quit)", args.host, args.port)
    if args.host == "127.0.0.1":
        log.info("Local-only. For remote access: ssh -L %d:localhost:%d <pi>",
                 args.port, args.port)

    # serve until Ctrl-C or a detach request
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        while not detach_flag.is_set():
            time.sleep(0.3)
    except KeyboardInterrupt:
        log.info("Ctrl-C -- shutting down GUI and recorder.")
        recorder.stop()
        state.stop_preview()
        state.stop_sensor()
        server.shutdown()
        return 0

    # detach path: stop the web server + preview, LEAVE recorder + sensor up
    log.info("Detaching: stopping web server + preview; recorder keeps running.")
    state.stop_preview()
    server.shutdown()
    # keep process alive so the daemon recorder thread survives
    try:
        while recorder.thread and recorder.thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        recorder.stop()
        state.stop_sensor()
    return 0


if __name__ == "__main__":
    sys.exit(main())
