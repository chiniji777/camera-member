# LAN Camera Viewer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a web-based LAN camera viewer supporting RTSP/ONVIF IP cameras and shared webcams, displaying up to 16 feeds in a grid.

**Architecture:** Python FastAPI server proxies RTSP streams via FFmpeg to HLS for the browser. A separate lightweight agent script shares webcams as MJPEG. A single-page HTML/JS frontend renders feeds in a configurable grid.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, FFmpeg, onvif-zeep, OpenCV, hls.js (CDN), vanilla HTML/CSS/JS

---

### Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `server.py` (minimal hello world)

**Step 1: Create requirements.txt**

```
fastapi
uvicorn[standard]
onvif-zeep
opencv-python
```

**Step 2: Create minimal server.py**

```python
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="LAN Camera Viewer")

@app.get("/api/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

**Step 3: Install dependencies**

Run: `pip install -r requirements.txt`

**Step 4: Verify server starts**

Run: `python server.py`
Then in another terminal: `curl http://localhost:8080/api/health`
Expected: `{"status":"ok"}`

---

### Task 2: Camera Manager — Data Model & CRUD

**Files:**
- Create: `camera_manager.py`

**Step 1: Write CameraManager class with CRUD and JSON persistence**

```python
import json
import uuid
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

CAMERAS_FILE = Path("cameras.json")

@dataclass
class Camera:
    id: str
    name: str
    url: str  # RTSP URL or MJPEG agent URL
    type: str  # "rtsp" or "mjpeg"
    enabled: bool = True

class CameraManager:
    def __init__(self):
        self.cameras: dict[str, Camera] = {}
        self._load()

    def _load(self):
        if CAMERAS_FILE.exists():
            data = json.loads(CAMERAS_FILE.read_text())
            for item in data:
                cam = Camera(**item)
                self.cameras[cam.id] = cam

    def _save(self):
        data = [asdict(cam) for cam in self.cameras.values()]
        CAMERAS_FILE.write_text(json.dumps(data, indent=2))

    def add(self, name: str, url: str, cam_type: str) -> Camera:
        cam_id = str(uuid.uuid4())[:8]
        cam = Camera(id=cam_id, name=name, url=url, type=cam_type)
        self.cameras[cam_id] = cam
        self._save()
        return cam

    def remove(self, cam_id: str) -> bool:
        if cam_id in self.cameras:
            del self.cameras[cam_id]
            self._save()
            return True
        return False

    def list_all(self) -> list[Camera]:
        return list(self.cameras.values())

    def get(self, cam_id: str) -> Optional[Camera]:
        return self.cameras.get(cam_id)
```

**Step 2: Quick manual test**

Run in Python REPL:
```python
from camera_manager import CameraManager
mgr = CameraManager()
cam = mgr.add("Test Cam", "rtsp://192.168.1.100:554/stream", "rtsp")
print(mgr.list_all())
mgr.remove(cam.id)
print(mgr.list_all())
```
Expected: camera added, listed, then removed. `cameras.json` created on disk.

---

### Task 3: FFmpeg Stream Manager

**Files:**
- Modify: `camera_manager.py` — add `StreamManager` class

**Step 1: Add StreamManager that spawns/kills FFmpeg processes**

Append to `camera_manager.py`:

```python
import subprocess
import shutil
import time
import threading
import logging

logger = logging.getLogger(__name__)

STREAMS_DIR = Path("streams")

class StreamManager:
    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}
        self._restart_counts: dict[str, int] = {}
        self._running = True
        STREAMS_DIR.mkdir(exist_ok=True)

    def start_stream(self, camera: Camera):
        if camera.type != "rtsp":
            return  # MJPEG cameras don't need FFmpeg

        stream_dir = STREAMS_DIR / camera.id
        stream_dir.mkdir(exist_ok=True)

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            logger.error("FFmpeg not found in PATH")
            return

        playlist = stream_dir / "stream.m3u8"

        cmd = [
            ffmpeg_path,
            "-rtsp_transport", "tcp",
            "-i", camera.url,
            "-c:v", "copy",
            "-c:a", "aac",
            "-f", "hls",
            "-hls_time", "3",
            "-hls_list_size", "3",
            "-hls_flags", "delete_segments",
            str(playlist)
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.processes[camera.id] = proc
        self._restart_counts[camera.id] = 0

        # Monitor thread for auto-restart
        t = threading.Thread(target=self._monitor, args=(camera,), daemon=True)
        t.start()

        logger.info(f"Started stream for {camera.name} ({camera.id})")

    def _monitor(self, camera: Camera):
        while self._running and camera.id in self.processes:
            proc = self.processes.get(camera.id)
            if proc is None:
                break
            ret = proc.poll()
            if ret is not None and self._running:
                count = self._restart_counts.get(camera.id, 0)
                delay = min(2 ** count, 30)
                logger.warning(f"FFmpeg for {camera.id} exited ({ret}), restarting in {delay}s")
                time.sleep(delay)
                self._restart_counts[camera.id] = count + 1
                if self._running and camera.id in self.processes:
                    del self.processes[camera.id]
                    self.start_stream(camera)
                break
            time.sleep(1)

    def stop_stream(self, cam_id: str):
        proc = self.processes.pop(cam_id, None)
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        # Clean up stream dir
        stream_dir = STREAMS_DIR / cam_id
        if stream_dir.exists():
            shutil.rmtree(stream_dir, ignore_errors=True)

    def stop_all(self):
        self._running = False
        for cam_id in list(self.processes.keys()):
            self.stop_stream(cam_id)
```

**Step 2: Verify FFmpeg is available**

Run: `ffmpeg -version`
Expected: FFmpeg version info. If not installed, user needs to install it.

---

### Task 4: ONVIF Discovery

**Files:**
- Create: `discovery.py`

**Step 1: Write ONVIF LAN scanner**

```python
import socket
import struct
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ONVIF_PROBE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
    xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
    xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
    xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
    <e:Header>
        <w:MessageID>uuid:84ede3de-7dec-11d0-c360-f01234567890</w:MessageID>
        <w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
        <w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
    </e:Header>
    <e:Body>
        <d:Probe>
            <d:Types>dn:NetworkVideoTransmitter</d:Types>
        </d:Probe>
    </e:Body>
</e:Envelope>"""

@dataclass
class DiscoveredCamera:
    address: str
    name: str
    xaddr: str  # ONVIF service URL

def discover_onvif_cameras(timeout: float = 3.0) -> list[DiscoveredCamera]:
    """Send WS-Discovery probe and collect ONVIF camera responses."""
    cameras = []

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(
        socket.IPPROTO_IP,
        socket.IP_MULTICAST_TTL,
        struct.pack("b", 2)
    )
    sock.settimeout(timeout)

    # WS-Discovery multicast address
    MULTICAST_ADDR = "239.255.255.250"
    MULTICAST_PORT = 3702

    try:
        sock.sendto(ONVIF_PROBE.encode(), (MULTICAST_ADDR, MULTICAST_PORT))

        while True:
            try:
                data, addr = sock.recvfrom(65535)
                response = data.decode(errors="ignore")

                # Extract XAddrs from response
                xaddr = _extract_xaddr(response)
                if xaddr:
                    cameras.append(DiscoveredCamera(
                        address=addr[0],
                        name=f"Camera at {addr[0]}",
                        xaddr=xaddr
                    ))
            except socket.timeout:
                break
    finally:
        sock.close()

    return cameras

def _extract_xaddr(xml_text: str) -> str:
    """Extract XAddrs URL from WS-Discovery response (simple parsing)."""
    import re
    match = re.search(r'<[^>]*XAddrs[^>]*>\s*(https?://[^\s<]+)', xml_text)
    return match.group(1) if match else ""

def get_rtsp_url_from_onvif(xaddr: str) -> str:
    """Try to get RTSP stream URL from ONVIF service address.

    Returns a best-guess RTSP URL based on the ONVIF address.
    For full ONVIF profile support, would need authentication.
    """
    # Extract IP and port from xaddr
    import re
    match = re.search(r'https?://([\d.]+)(?::(\d+))?', xaddr)
    if match:
        ip = match.group(1)
        # Common RTSP paths for various manufacturers
        return f"rtsp://{ip}:554/stream1"
    return ""
```

**Step 2: Test discovery manually**

Run in Python REPL:
```python
from discovery import discover_onvif_cameras
cams = discover_onvif_cameras(timeout=5)
print(f"Found {len(cams)} cameras")
for c in cams:
    print(f"  {c.address}: {c.xaddr}")
```
Expected: Lists any ONVIF cameras on the LAN (may be 0 if none present).

---

### Task 5: Wire Up API Endpoints in server.py

**Files:**
- Modify: `server.py` — full API implementation

**Step 1: Rewrite server.py with all endpoints**

```python
import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from camera_manager import CameraManager, StreamManager
from discovery import discover_onvif_cameras, get_rtsp_url_from_onvif

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

camera_mgr = CameraManager()
stream_mgr = StreamManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: start streams for all saved cameras
    for cam in camera_mgr.list_all():
        if cam.enabled:
            stream_mgr.start_stream(cam)
    yield
    # Shutdown: stop all streams
    stream_mgr.stop_all()

app = FastAPI(title="LAN Camera Viewer", lifespan=lifespan)

# --- Models ---
class AddCameraRequest(BaseModel):
    name: str
    url: str
    type: str = "rtsp"  # "rtsp" or "mjpeg"

# --- API ---
@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.get("/api/cameras")
async def list_cameras():
    return [
        {
            "id": cam.id,
            "name": cam.name,
            "url": cam.url,
            "type": cam.type,
            "enabled": cam.enabled,
            "stream_url": f"/streams/{cam.id}/stream.m3u8" if cam.type == "rtsp" else cam.url
        }
        for cam in camera_mgr.list_all()
    ]

@app.post("/api/cameras")
async def add_camera(req: AddCameraRequest):
    cam = camera_mgr.add(req.name, req.url, req.type)
    if cam.type == "rtsp":
        stream_mgr.start_stream(cam)
    return {"id": cam.id, "name": cam.name}

@app.delete("/api/cameras/{cam_id}")
async def remove_camera(cam_id: str):
    stream_mgr.stop_stream(cam_id)
    if not camera_mgr.remove(cam_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"ok": True}

@app.post("/api/discover")
async def discover():
    found = discover_onvif_cameras(timeout=5)
    results = []
    for cam in found:
        rtsp_url = get_rtsp_url_from_onvif(cam.xaddr)
        results.append({
            "address": cam.address,
            "name": cam.name,
            "xaddr": cam.xaddr,
            "rtsp_url": rtsp_url
        })
    return results

# --- HLS Stream serving ---
STREAMS_DIR = Path("streams")

@app.get("/streams/{cam_id}/{filename}")
async def serve_stream(cam_id: str, filename: str):
    file_path = STREAMS_DIR / cam_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Stream not ready")

    content_type = "application/vnd.apple.mpegurl" if filename.endswith(".m3u8") else "video/mp2t"
    return FileResponse(
        file_path,
        media_type=content_type,
        headers={"Cache-Control": "no-cache, no-store"}
    )

# --- Static files (Web UI) ---
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

@app.get("/")
async def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Web UI not yet built. Place index.html in static/"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

**Step 2: Test API endpoints**

Run: `python server.py` and in another terminal:
```bash
# Health check
curl http://localhost:8080/api/health

# List cameras (empty)
curl http://localhost:8080/api/cameras

# Add a test camera
curl -X POST http://localhost:8080/api/cameras -H "Content-Type: application/json" -d '{"name":"Test","url":"rtsp://test","type":"rtsp"}'

# List again
curl http://localhost:8080/api/cameras

# Discover
curl -X POST http://localhost:8080/api/discover
```

---

### Task 6: Webcam Agent

**Files:**
- Create: `agent.py`

**Step 1: Write the webcam agent**

```python
"""
Webcam Agent — Run on any PC to share its webcam over the LAN.

Usage: python agent.py [--port 8554] [--camera 0]

Other devices on the LAN can view the stream at:
    http://<this-pc-ip>:<port>/video
"""
import argparse
import socket
import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MJPEGHandler(BaseHTTPRequestHandler):
    camera = None
    lock = threading.Lock()

    def do_GET(self):
        if self.path == "/video":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                while True:
                    with MJPEGHandler.lock:
                        ret, frame = MJPEGHandler.camera.read()
                    if not ret:
                        continue
                    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    data = jpeg.tobytes()
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Webcam Agent</h1>")
            self.wfile.write(b'<img src="/video" style="max-width:100%"></body></html>')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress per-request logs

def get_local_ip():
    """Get the LAN IP of this machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()

def main():
    parser = argparse.ArgumentParser(description="Webcam Agent — share webcam over LAN")
    parser.add_argument("--port", type=int, default=8554, help="HTTP port (default: 8554)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {args.camera}")
        return

    MJPEGHandler.camera = cap

    local_ip = get_local_ip()
    server = HTTPServer(("0.0.0.0", args.port), MJPEGHandler)

    logger.info(f"Webcam agent running")
    logger.info(f"  Local:   http://localhost:{args.port}/video")
    logger.info(f"  Network: http://{local_ip}:{args.port}/video")
    logger.info(f"Add this URL to the camera viewer to see this feed.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        server.shutdown()

if __name__ == "__main__":
    main()
```

**Step 2: Test agent**

Run: `python agent.py --port 8554 --camera 0`
Open browser: `http://localhost:8554/video`
Expected: Live webcam feed visible in browser.

---

### Task 7: Web UI

**Files:**
- Create: `static/index.html`

**Step 1: Write the complete web UI**

This is the largest single file. It includes all HTML, CSS, and JS in one file. Key features:
- Dark theme
- Grid selector toolbar (1x1, 2x2, 3x3, 4x4)
- Add camera modal (name, URL, type dropdown)
- Discover cameras button
- Click-to-fullscreen on any tile
- Auto-reconnect with offline overlay
- hls.js for RTSP/HLS streams, native `<img>` for MJPEG

The full HTML file will be ~400 lines of self-contained code.

**Step 2: Test the full stack**

1. Start server: `python server.py`
2. Open `http://localhost:8080` in browser
3. Click "Add Camera" and add an MJPEG or RTSP URL
4. Verify feed appears in grid
5. Test grid size buttons
6. Test fullscreen click
7. Test remove camera

---

### Task 8: Integration Testing & Polish

**Step 1: Test with webcam agent**

Terminal 1: `python agent.py --port 8554`
Terminal 2: `python server.py`
Browser: Add camera with URL `http://localhost:8554/video`, type MJPEG
Expected: Webcam feed appears in grid.

**Step 2: Test with RTSP (if available)**

Add an RTSP camera URL through the UI.
Expected: Feed appears after ~5-10 seconds (HLS buffering).

**Step 3: Test discovery**

Click "Discover" button.
Expected: Any ONVIF cameras on LAN listed (or empty list if none).

**Step 4: Test persistence**

1. Add 2 cameras
2. Stop and restart server
3. Cameras should auto-load from cameras.json

**Step 5: Test error handling**

1. Add camera with bad URL
2. Verify "Offline" overlay appears
3. Stop webcam agent while viewing — verify reconnect behavior

---

## Summary

| Task | Component | Estimated Complexity |
|------|-----------|---------------------|
| 1 | Scaffolding | Simple |
| 2 | Camera Manager CRUD | Simple |
| 3 | FFmpeg Stream Manager | Medium |
| 4 | ONVIF Discovery | Medium |
| 5 | API Endpoints | Medium |
| 6 | Webcam Agent | Simple |
| 7 | Web UI | Large |
| 8 | Integration Testing | Medium |

Build order is sequential — each task builds on the previous.
