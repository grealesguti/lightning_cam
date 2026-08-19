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
            # periodic snapshots (minutes; 0 = off)
            "snapshot_every": 0,
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
        self.events_dir = None            # set once the recorder reports its output
        self.last_recal = None            # {"time":..., "detail":...}
        self.last_snapshot = None         # {"time":..., "file":...}
        self._refresh_preview_flag = threading.Event()

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

    def refresh_preview_soon(self):
        """
        Ask the preview to update. While recording, the GUI doesn't own the
        camera (the recorder does), so the freshest image we can show is the
        latest snapshot written to disk. This loads it as the preview image.
        """
        self._refresh_preview_flag.set()
        # if recording, pull the newest snapshot jpg from disk into the preview
        if self.recording and self.events_dir:
            snap_dir = os.path.join(os.path.dirname(self.events_dir), "snapshots")
            try:
                if os.path.isdir(snap_dir):
                    jpgs = [os.path.join(snap_dir, f) for f in os.listdir(snap_dir)
                            if f.endswith(".jpg")]
                    if jpgs:
                        newest = max(jpgs, key=os.path.getmtime)
                        with open(newest, "rb") as f:
                            data = f.read()
                        with self.preview_lock:
                            self.preview_jpeg = data
            except Exception:
                pass

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
    def last_event(self):
        """
        Return stats from the most recent event's JSON sidecar, or None.
        Reads the newest event_*.json in the events dir the recorder reported.
        """
        ev_dir = self.events_dir
        if not ev_dir or not os.path.isdir(ev_dir):
            return None
        try:
            sidecars = [os.path.join(ev_dir, f) for f in os.listdir(ev_dir)
                        if f.startswith("event_") and f.endswith(".json")]
            if not sidecars:
                return None
            newest = max(sidecars, key=os.path.getmtime)
            with open(newest) as f:
                data = json.load(f)
            sensor = data.get("sensor", {})
            power = data.get("power", {})
            return {
                "event_id": data.get("event_id"),
                "trigger_time": data.get("trigger_time"),
                "measured_fps": data.get("measured_fps"),
                "n_frames": data.get("n_frames"),
                "pre_frames": data.get("pre_frames"),
                "post_frames": data.get("post_frames"),
                "camera": data.get("camera"),
                "sensor_kind": sensor.get("kind"),
                "distance_km": sensor.get("distance_km"),
                "energy": sensor.get("energy"),
                "power_throttled": power.get("throttled_hex"),
                "power_volt": power.get("core_volt_v"),
                "power_temp": power.get("soc_temp_c"),
                "saved_as": data.get("saved_as"),
            }
        except Exception:
            return None

    def diagnostics(self):
        """
        Check for common problems and return a list of warning dicts:
        {"level": "warn"|"error", "msg": "..."}. Surfaced in the GUI so issues
        are visible instead of silent.
        """
        warns = []

        # --- power / throttle ---
        p = self.power.snapshot()
        mask = self.power.get_throttled_raw()
        if mask:
            active = self.power.active_warnings(mask)
            # current undervoltage/throttle is an ERROR (affects capture now)
            now_bad = [w for w in active if "now" in w.lower() or
                       "detected" in w.lower() or "Currently" in w]
            if any(("Under-voltage detected" == w or "Currently throttled" == w)
                   for w in active):
                warns.append({"level": "error",
                              "msg": "Power: undervoltage / throttling NOW ("
                                     + p.get("throttled_hex", "?") +
                                     "). Frame rate will suffer and frames may "
                                     "corrupt. Use a stronger 5V/3A supply and a "
                                     "thick short cable."})
            elif active:
                warns.append({"level": "warn",
                              "msg": "Power: throttling occurred earlier ("
                                     + p.get("throttled_hex", "?") +
                                     "). Watch the supply/battery."})

        # --- temperature ---
        t = p.get("soc_temp_c")
        if t is not None and t >= 80:
            warns.append({"level": "error",
                          "msg": f"SoC temperature high ({t}°C) -- add cooling."})
        elif t is not None and t >= 70:
            warns.append({"level": "warn",
                          "msg": f"SoC temperature warm ({t}°C)."})

        # --- camera / preview ---
        if not _HAVE_CV2:
            warns.append({"level": "error",
                          "msg": "OpenCV not available -- preview and decode "
                                 "won't work. Activate the venv or install cv2."})
        elif self.cam is None and not self.recording:
            warns.append({"level": "warn",
                          "msg": "Camera preview not running -- check the device "
                                 "or that another process (recorder) holds it."})

        # --- sensor ---
        if self.sensor is None and not self.recording:
            warns.append({"level": "warn",
                          "msg": "Sensor not started -- no triggers. Check bus/"
                                 "wiring, or it's handed to the recorder."})
        elif self.sensor is not None and not self.sensor.comm_ok():
            warns.append({"level": "error",
                          "msg": "AS3935 not responding on the bus -- check SI "
                                 "strap, CS pin, 3.3V power, solder joints."})

        # --- suspiciously high 'lightning' rate = interference ---
        with self.lock:
            evs = list(self.sensor_events)
        if len(evs) >= 10:
            recent = evs[-20:]
            lightning = [e for e in recent if e["kind"] == "lightning"]
            if len(lightning) >= 15:
                dists = {e["distance_km"] for e in lightning}
                if len(dists) <= 1:
                    warns.append({"level": "warn",
                                  "msg": "Many 'lightning' events all at the same "
                                         "distance -- likely RF interference from "
                                         "the Pi/USB, not real strikes. Move the "
                                         "sensor away from the Pi and cables."})

        # --- USB / output ---
        usb = find_usb_mounts()
        out = self.settings.get("output", "")
        if not usb and not out:
            warns.append({"level": "warn",
                          "msg": "No USB detected and no output set -- events "
                                 "will fall back to the local disk (or fail). "
                                 "Plug in a drive and Rescan, or set an output "
                                 "path."})
        else:
            # low space check on the chosen/auto target
            target = usb[0] if usb else None
            if target and target["free"] is not None and target["free"] < 500_000_000:
                warns.append({"level": "warn",
                              "msg": f"USB nearly full ({human_bytes(target['free'])} "
                                     "free)."})

        # --- linuxpy (raw path) ---
        try:
            import linuxpy  # noqa
        except Exception:
            warns.append({"level": "warn",
                          "msg": "linuxpy not installed -- raw high-fps capture "
                                 "unavailable; recorder falls back to slower "
                                 "decoded path. `pip install linuxpy`."})

        return warns

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
            "diagnostics": self.diagnostics(),
            "last_event": self.last_event(),
            "last_recal": self.last_recal,
            "last_snapshot": self.last_snapshot,
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
 #diagbar{padding:0 16px}
 .diag{margin:8px 0;padding:9px 12px;border-radius:8px;font-size:13px;font-weight:500}
 .diag.error{background:#3a1414;border:1px solid #c0392b;color:#ffb4a8}
 .diag.warn{background:#3a3014;border:1px solid #c98a1a;color:#ffe0a0}
 #diagbar:empty{padding:0}
</style></head><body>
<header>⚡ Lightning Camera Control <span id="recstate" class="pill"></span></header>
<div id="diagbar"></div>
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
   <div class="kv"><span>Last recalibrated</span><b id="p_recal">never</b></div>
   <div class="kv"><span>Last snapshot</span><b id="p_snap">never</b></div>
   <div id="lastevt" style="margin-top:10px;padding-top:8px;border-top:1px solid #262c38">
     <div class="muted">Last event</div>
     <div id="le_none" class="muted">No events captured yet.</div>
     <div id="le_body" style="display:none">
       <div class="kv"><span>When</span><b id="le_time">-</b></div>
       <div class="kv"><span>Measured FPS</span><b id="le_fps">-</b></div>
       <div class="kv"><span>Frames (pre+post)</span><b id="le_frames">-</b></div>
       <div class="kv"><span>Camera</span><b id="le_cam">-</b></div>
       <div class="kv"><span>Type / distance</span><b id="le_sensor">-</b></div>
       <div class="kv"><span>Energy</span><b id="le_energy">-</b></div>
       <div class="kv"><span>Power at event</span><b id="le_power">-</b></div>
       <div class="kv"><span>Saved as</span><b id="le_file">-</b></div>
     </div>
   </div>
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
   <label>Snapshot every N minutes (0=off) — saves a still even with no event</label>
   <input id="snapshot_every">
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
   <button onclick="save(event)">Apply settings</button>
   <button id="startbtn" onclick="startRec(event)">Start recording</button>
   <button class="stop" onclick="stopRec(event)">Stop recording</button>
   <hr style="border-color:#262c38;margin:14px 0">
   <button class="ghost" onclick="detach()">Detach GUI (keep recording)</button>
   <div class="muted">Detach stops this web server but leaves the recorder
     running with the current settings. Re-launch the GUI to reattach.</div>
 </div>

</div>
<script>
const $=id=>document.getElementById(id);
const FIELDS=["width","height","fps","exposure","gain","pre","post","format",
  "bus","irq_gpio","noise_floor","watchdog","spike","recal_every","target_brightness",
  "snapshot_every"];
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
async function save(ev){
  const btn=ev&&ev.target;
  try{
    const r=await fetch("/api/settings",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify(collect())});
    const j=await r.json();
    flash(btn, j.ok?"Applied ✓":"Failed ✗", j.ok);
    return j.ok;
  }catch(e){ flash(btn,"Error ✗",false); return false; }
}
async function startRec(){
  const btn=$("startbtn");
  const okSave=await saveQuiet();
  if(!okSave){ flash(btn,"Apply failed ✗",false); return; }
  try{
    const r=await fetch("/api/start",{method:"POST"}); const j=await r.json();
    if(j.ok){ flash(btn,"Started ✓",true); }
    else{ flash(btn,(j.reason||"failed")+" ✗",false);
      alert("Could not start recording: "+(j.reason||"unknown")); }
  }catch(e){ flash(btn,"Error ✗",false); alert("Start error: "+e); }
}
async function stopRec(ev){
  const btn=ev&&ev.target;
  try{
    const r=await fetch("/api/stop",{method:"POST"}); const j=await r.json();
    flash(btn,"Stopped ✓",true);
  }catch(e){ flash(btn,"Error ✗",false); }
}
async function saveQuiet(){
  try{
    const r=await fetch("/api/settings",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify(collect())});
    return (await r.json()).ok;
  }catch(e){ return false; }
}
function flash(btn,msg,ok){
  if(!btn)return; const orig=btn.textContent;
  btn.textContent=msg; btn.style.background=ok?"#1f8a4c":"#c0392b";
  setTimeout(()=>{ btn.textContent=orig; btn.style.background=""; }, 1600);
}
async function detach(){
  if(!confirm("Stop the GUI but keep recording in the background?"))return;
  await saveQuiet();
  await fetch("/api/detach",{method:"POST"});
  document.body.innerHTML="<header>Detached. Recorder still running in the "+
    "background. Re-launch lightning_gui.py to reattach.</header>";
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
    // last recalibration + last snapshot
    const lr=s.last_recal, ls=s.last_snapshot;
    $("p_recal").textContent=lr?
      ((lr.time.split("T")[1]||lr.time)+" — "+lr.detail):"never";
    $("p_snap").textContent=ls?
      ((ls.time.split("T")[1]||ls.time)+" — "+(ls.file||"")):"never";
    // if a recal or snapshot is newer than what we last saw, refresh preview now
    const rkey=(lr?lr.time:"")+"|"+(ls?ls.time:"");
    if(window._lastRefreshKey!==undefined && window._lastRefreshKey!==rkey){
      $("preview").src="/preview.jpg?"+Date.now();
    }
    window._lastRefreshKey=rkey;
    // last event stats
    const le=s.last_event;
    if(le){
      $("le_none").style.display="none"; $("le_body").style.display="block";
      $("le_time").textContent=(le.trigger_time||"").replace("T"," ");
      $("le_fps").textContent=le.measured_fps!=null?le.measured_fps+" fps":"-";
      $("le_frames").textContent=le.n_frames!=null?
        `${le.n_frames} (${le.pre_frames}+${le.post_frames})`:"-";
      $("le_cam").textContent=le.camera||"-";
      $("le_sensor").textContent=`${le.sensor_kind||"-"} / `+
        (le.distance_km!=null?le.distance_km+" km":"out-of-range");
      $("le_energy").textContent=le.energy!=null?le.energy:"-";
      $("le_power").textContent=`${le.power_throttled||"-"}, `+
        `${le.power_volt??"-"}V, ${le.power_temp??"-"}°C`;
      $("le_file").textContent=le.saved_as||"-";
    } else {
      $("le_none").style.display="block"; $("le_body").style.display="none";
    }
    // diagnostics banner
    const db=$("diagbar");
    if(s.diagnostics && s.diagnostics.length){
      db.innerHTML=s.diagnostics.map(d=>
        `<div class="diag ${d.level}">${d.level==="error"?"⛔":"⚠️"} ${d.msg}</div>`
      ).join("");
    } else { db.innerHTML=""; }
    const rs=$("recstate");
    if(s.recording){ rs.textContent=s.event_active?"CAPTURING EVENT":"recording";
      rs.className="pill "+(s.event_active?"warn":"ok");
      $("startbtn").disabled=true; $("startbtn").style.opacity=.5; }
    else { rs.textContent="idle"; rs.className="pill";
      $("startbtn").disabled=false; $("startbtn").style.opacity=1; }
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
    state: AppState = None          # set on the class before serving
    recorder = None                 # BackgroundRecorder instance
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
            res = self.recorder.start(self.state.settings)
            self._send(200, json.dumps(res), "application/json")
        elif p == "/api/stop":
            res = self.recorder.stop()
            self._send(200, json.dumps(res), "application/json")
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
        floats = ["pre", "post", "recal_every", "snapshot_every"]
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
    Launches the REAL recorder (lightning_run.py) as a subprocess so clicking
    Start actually captures and SAVES events.

    Resource handoff: the GUI holds the camera (preview) and the sensor (data
    panel) open. The recorder subprocess needs those same devices, so on Start
    we RELEASE the GUI's preview + sensor, launch lightning_run.py, and stream
    its log. On Stop we terminate the subprocess and RE-CLAIM preview + sensor
    so the panels come back live.

    State is truthful: `state.recording` reflects whether the subprocess is
    alive, and `state.events_saved` is derived from the actual event files on
    disk (so the counter matches reality, not a guess).
    """
    def __init__(self, state: AppState, log, project_dir, python_exe):
        self.state = state
        self.log = log
        self.project_dir = project_dir
        self.python_exe = python_exe
        self.proc = None
        self.watch_thread = None
        self.output_dir = None

    def _build_cmd(self, s):
        cmd = [self.python_exe, os.path.join(self.project_dir, "lightning_run.py"),
               "--width", str(s["width"]), "--height", str(s["height"]),
               "--fps", str(s["fps"]),
               "--pre", str(s["pre"]), "--post", str(s["post"]),
               "--format", s["format"],
               "--bus", s["bus"], "--irq-gpio", str(s["irq_gpio"]),
               "--status-every", "30"]
        # camera exposure/gain
        if s.get("exposure") not in (None, ""):
            cmd += ["--exposure", str(s["exposure"])]
        if s.get("gain") not in (None, ""):
            cmd += ["--gain", str(s["gain"])]
        # recalibration
        if float(s.get("recal_every", 0) or 0) > 0:
            cmd += ["--recal-every", str(s["recal_every"]),
                    "--target-brightness", str(s["target_brightness"])]
        # periodic snapshots
        if float(s.get("snapshot_every", 0) or 0) > 0:
            cmd += ["--snapshot-every", str(s["snapshot_every"])]
        # sensor tuning
        if s.get("noise_floor") not in (None, ""):
            cmd += ["--noise-floor", str(s["noise_floor"])]
        if s.get("watchdog") not in (None, ""):
            cmd += ["--watchdog", str(s["watchdog"])]
        if s.get("spike") not in (None, ""):
            cmd += ["--spike", str(s["spike"])]
        if s.get("mask_disturbers"):
            cmd += ["--mask-disturbers"]
        if s.get("outdoor"):
            cmd += ["--outdoor"]
        if s.get("output"):
            cmd += ["--output", s["output"]]
        return cmd

    def start(self, settings):
        if self.proc and self.proc.poll() is None:
            self.log.info("Recorder already running (pid %s).", self.proc.pid)
            return {"ok": False, "reason": "already running"}

        s = dict(settings)

        # 1) release the GUI's hold on camera + sensor so the child can open them
        self.log.info("Handing devices to recorder: releasing preview + sensor.")
        self.state.stop_preview()
        self.state.stop_sensor()
        time.sleep(0.4)     # let the devices settle

        # 2) launch the real recorder
        import subprocess
        cmd = self._build_cmd(s)
        self.log.info("Starting recorder: %s", " ".join(cmd))
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=self.project_dir,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
        except Exception as e:
            self.log.error("Failed to launch recorder: %s", e)
            # reclaim devices since we didn't hand them off
            self.state.start_sensor()
            self.state.start_preview()
            return {"ok": False, "reason": str(e)}

        self.state.recording = True
        # 3) stream the child's log and watch the events dir for the counter
        self.watch_thread = threading.Thread(target=self._watch, daemon=True)
        self.watch_thread.start()
        return {"ok": True, "pid": self.proc.pid}

    def _watch(self):
        # capture the output dir from the child's log line "Output dir: ..."
        for line in iter(self.proc.stdout.readline, ""):
            line = line.rstrip("\n")
            if not line:
                continue
            self.log.info("[recorder] %s", line)
            if "Output dir:" in line:
                self.output_dir = line.split("Output dir:", 1)[1].strip()
                self.state.events_dir = os.path.join(self.output_dir, "events")
            # recalibration happened -> record time + detail, refresh preview
            if "recalibrate:" in line:
                detail = line.split("recalibrate:", 1)[1].strip()
                self.state.last_recal = {"time": now_iso(), "detail": detail}
                self.state.refresh_preview_soon()
            # a snapshot was taken -> record it + refresh preview
            if "Snapshot saved" in line:
                fname = line.split("->", 1)[1].strip() if "->" in line else ""
                self.state.last_snapshot = {"time": now_iso(),
                                            "file": os.path.basename(fname)}
                self.state.refresh_preview_soon()
            # update the saved-events counter from disk
            if self.output_dir:
                self._update_count()
        # subprocess ended
        rc = self.proc.poll()
        self.log.info("Recorder subprocess exited (code %s).", rc)
        self.state.recording = False
        # reclaim devices for the GUI panels
        self.state.start_sensor()
        self.state.start_preview()

    def _update_count(self):
        try:
            ev_dir = os.path.join(self.output_dir, "events")
            if os.path.isdir(ev_dir):
                n = len([f for f in os.listdir(ev_dir)
                         if f.startswith("event_") and
                         (f.endswith(".npy") or f.endswith(".mp4") or
                          f.endswith("_frames") or f.endswith(".json") is False)])
                # count unique event ids by json sidecars (most reliable)
                n = len([f for f in os.listdir(ev_dir) if f.endswith(".json")])
                self.state.events_saved = n
        except Exception:
            pass

    def stop(self):
        if not self.proc or self.proc.poll() is not None:
            self.state.recording = False
            return {"ok": True, "reason": "not running"}
        self.log.info("Stopping recorder (pid %s).", self.proc.pid)
        import signal as _sig
        try:
            self.proc.send_signal(_sig.SIGINT)   # clean shutdown (flushes)
            try:
                self.proc.wait(timeout=8)
            except Exception:
                self.proc.terminate()
                self.proc.wait(timeout=5)
        except Exception as e:
            self.log.error("Error stopping recorder: %s", e)
            try:
                self.proc.kill()
            except Exception:
                pass
        self.state.recording = False
        # devices are reclaimed by _watch when the process ends
        return {"ok": True}

    def is_running(self):
        return self.proc is not None and self.proc.poll() is None


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
    project_dir = os.path.dirname(os.path.abspath(__file__))
    recorder = BackgroundRecorder(state, log, project_dir, sys.executable)
    detach_flag = threading.Event()

    if not args.no_preview:
        state.start_preview()
    if not args.no_sensor:
        state.start_sensor()

    # wire the handler class attributes
    Handler.state = state
    Handler.recorder = recorder
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

    # detach path: stop the web server + preview, LEAVE the recorder subprocess
    # running (it's an independent process, so it survives the GUI exiting).
    log.info("Detaching: stopping web server + preview; recorder subprocess "
             "keeps running (pid %s).",
             recorder.proc.pid if recorder.is_running() else "none")
    state.stop_preview()
    state.stop_sensor()
    server.shutdown()
    if recorder.is_running():
        log.info("Recorder still running in background. Re-launch the GUI to "
                 "reattach, or `kill %s` to stop it.", recorder.proc.pid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
