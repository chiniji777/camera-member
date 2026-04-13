"""LAN Camera Viewer — Main Server

Start with:  python server.py
Open:         http://localhost:8080
"""
import json
import asyncio
import uvicorn
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, Response
from pydantic import BaseModel
from pathlib import Path
from camera_manager import CameraManager, StreamManager, AudioStreamManager, auto_resolve_cameras
from ezviz_manager import EzvizTokenManager
from discovery import discover_onvif_cameras, get_rtsp_url_from_onvif
from face_manager import FaceManager, FaceDetector
from smart_home import SmartHomeScanner
from layout_manager import LayoutManager
from motion_detector import MotionDetector
from self_learn import SelfLearnPipeline
from gate_automation import GateAutomation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

camera_mgr = CameraManager()
ezviz_mgr = EzvizTokenManager()
stream_mgr = StreamManager()
stream_mgr.set_camera_manager(camera_mgr)
stream_mgr.set_ezviz_manager(ezviz_mgr)
audio_mgr = AudioStreamManager()
face_mgr = FaceManager()
face_detector = FaceDetector(stream_mgr, face_mgr)
smart_home = SmartHomeScanner()
layout_mgr = LayoutManager()

# Link layout manager to face detector for detection toggle/sensitivity
face_detector.set_layout_manager(layout_mgr)

# Motion detector — wired into stream manager
motion_detector = MotionDetector(
    sensitivity=layout_mgr.get_sensitivity("motion"),
)
self_learn = SelfLearnPipeline()
stream_mgr.motion_detector = motion_detector

# WebSocket clients for motion events
_motion_clients: list[WebSocket] = []
_motion_lock = asyncio.Lock()


async def _broadcast_motion(event: dict):
    msg = json.dumps(event)
    async with _motion_lock:
        dead = []
        for ws in _motion_clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _motion_clients.remove(ws)


def _on_motion(cam_id: str, intensity: float):
    """Push motion event to all WebSocket clients (called from sync thread)."""
    import time
    event = {"type": "motion", "camera_id": cam_id,
             "intensity": round(intensity, 4), "ts": time.time()}
    try:
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(asyncio.ensure_future, _broadcast_motion(event))
    except RuntimeError:
        pass  # no event loop yet


motion_detector.on_motion(_on_motion)


# --- Voice Command: Gate control via Smart Home API ---
SMART_HOME_API = "https://home.thinkfirstconsult.com"
SMART_HOME_API_KEY = "e3356486972de7575891ab81a5d26d32d101f3b6f812dbe5"
GATE_DEVICE_ID = "tuya_eb1ded1455fd293487dapn"


def _send_gate_command(channel: int, label: str):
    """Send gate command to smart home API."""
    import requests
    channels = [
        {"outlet": 0, "power": channel == 0},
        {"outlet": 1, "power": channel == 1},
        {"outlet": 2, "power": channel == 2},
    ]
    try:
        resp = requests.post(
            f"{SMART_HOME_API}/api/devices/{GATE_DEVICE_ID}/command",
            json={"commands": {"channels": channels}},
            headers={"x-api-key": SMART_HOME_API_KEY},
            timeout=10,
            verify=True,
        )
        logger.info(f"[Gate] {label} → {resp.status_code}: {resp.text[:100]}")
    except Exception as e:
        logger.error(f"[Gate] Failed to send command: {e}")


gate_auto = GateAutomation(on_gate_command=_send_gate_command)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-resolve camera IPs if they changed (e.g. after power outage / DHCP)
    resolved = await asyncio.to_thread(auto_resolve_cameras, camera_mgr)
    if resolved:
        logger.info(f"Auto-resolved {resolved} camera IP(s)")

    for cam in camera_mgr.list_all():
        if cam.enabled:
            stream_mgr.start_stream(cam)
            if cam.type == "rtsp":
                audio_mgr.start_audio(cam)
    logger.info(f"Loaded {len(camera_mgr.cameras)} camera(s)")

    # Start face detection on all active cameras
    active_ids = [cam.id for cam in camera_mgr.list_all() if cam.enabled]
    if active_ids:
        face_detector.start(active_ids)
        logger.info(f"Face detection started on {len(active_ids)} camera(s)")

    # Enable motion detection based on saved setting
    if layout_mgr.is_detection_enabled("motion"):
        motion_detector.set_enabled(True)
        logger.info("Motion detection enabled at startup")

    # Start gate automation on first enabled RTSP camera
    for cam in camera_mgr.list_all():
        if cam.enabled and cam.type == "rtsp":
            gate_auto.start(cam.id)
            break

    yield

    gate_auto.stop()
    face_detector.stop()
    stream_mgr.stop_all()
    audio_mgr.stop_all()
    logger.info("All streams stopped")


app = FastAPI(title="LAN Camera Viewer", lifespan=lifespan)

# Allow embedding in iframe from arra-office and CORS for API calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3456"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class AllowIframeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Remove X-Frame-Options to allow iframe embedding
        if "x-frame-options" in response.headers:
            del response.headers["x-frame-options"]
        return response

app.add_middleware(AllowIframeMiddleware)

STATIC_DIR = Path("static")


# --- Models ---

class AddCameraRequest(BaseModel):
    name: str
    url: str
    type: str = "rtsp"


# --- API Endpoints ---

@app.get("/api/health")
async def health():
    from datetime import datetime
    return {"status": "ok", "server_time": datetime.now().astimezone().isoformat()}


@app.get("/api/cameras")
async def list_cameras():
    result = []
    for cam in camera_mgr.list_all():
        info = {
            "id": cam.id,
            "name": cam.name,
            "url": cam.url,
            "type": cam.type,
            "enabled": cam.enabled,
        }
        if cam.type == "ezviz":
            info["stream_url"] = cam.url  # ezopen:// URL
            info["ezviz_serial"] = cam.ezviz_serial
            info["ezviz_channel"] = cam.ezviz_channel
        else:
            info["stream_url"] = (
                f"/streams/{cam.id}/live" if cam.type == "rtsp" else cam.url
            )
        # Expose quality info for cameras with switchable channels
        if cam.type == "rtsp" and "/Streaming/Channels/" in cam.url:
            info["quality"] = "high" if "/Channels/101" in cam.url else "low"
            info["can_switch_quality"] = True
        result.append(info)
    return result


@app.post("/api/cameras")
async def add_camera(req: AddCameraRequest):
    cam = camera_mgr.add(req.name, req.url, req.type)
    if cam.type == "rtsp":
        stream_mgr.start_stream(cam)
        audio_mgr.start_audio(cam)
        face_detector.add_camera(cam.id)
    logger.info(f"Added camera: {cam.name} ({cam.type}) -> {cam.url}")
    return {"id": cam.id, "name": cam.name}


@app.get("/api/cameras/detailed")
async def list_cameras_detailed():
    """Return cameras with extra metadata (host, model, ptz) for frontend."""
    result = []
    for cam in camera_mgr.list_all():
        info = {
            "id": cam.id,
            "name": cam.name,
            "type": cam.type,
            "enabled": cam.enabled,
            "ptz": False,
        }
        if cam.type == "ezviz":
            info["stream_url"] = f"/streams/{cam.id}/live"
            info["ezviz_serial"] = cam.ezviz_serial
            info["ezviz_channel"] = cam.ezviz_channel
            info["host"] = ""
            info["model"] = "EZVIZ C6N"
            info["ptz"] = True  # C6N has PTZ
        else:
            info["stream_url"] = f"/streams/{cam.id}/live" if cam.type == "rtsp" else cam.url
            info["url"] = cam.url
            # Parse host from RTSP URL
            try:
                from urllib.parse import urlparse
                parsed = urlparse(cam.url)
                info["host"] = parsed.hostname or ""
            except Exception:
                info["host"] = ""
            if cam.type == "rtsp" and "/Streaming/Channels/" in cam.url:
                info["quality"] = "high" if "/Channels/101" in cam.url else "low"
                info["can_switch_quality"] = True
        result.append(info)
    return result


@app.get("/api/ezviz/token")
async def ezviz_token():
    """Return a valid EZVIZ access token for EZUIKit player."""
    token_info = ezviz_mgr.get_token()
    if not token_info:
        raise HTTPException(503, "Failed to get EZVIZ token")
    return token_info


@app.get("/api/ezviz/hls/{cam_id}")
async def ezviz_hls(cam_id: str, quality: int = 1):
    """Get HLS live stream URL for an EZVIZ camera.
    quality: 1=fluent, 2=HD, 3=super HD"""
    cam = camera_mgr.get(cam_id)
    if not cam or cam.type != "ezviz":
        raise HTTPException(404, "EZVIZ camera not found")

    token_info = ezviz_mgr.get_token()
    if not token_info:
        raise HTTPException(503, "Failed to get EZVIZ token")

    import requests as req_lib
    resp = req_lib.post(
        "https://isgpopen.ezvizlife.com/api/lapp/v2/live/address/get",
        data={
            "accessToken": token_info["accessToken"],
            "deviceSerial": cam.ezviz_serial or cam_id,
            "channelNo": cam.ezviz_channel or 1,
            "protocol": 2,
            "quality": quality,
        },
        timeout=10,
    )
    data = resp.json()
    if data.get("code") != "200":
        raise HTTPException(502, f"EZVIZ API error: {data.get('msg', 'unknown')}")
    return {"url": data["data"]["url"], "expireTime": data["data"].get("expireTime")}


class EzvizPtzRequest(BaseModel):
    direction: str
    speed: int = 50


# Map direction names to EZVIZ PTZ command codes
EZVIZ_PTZ_MAP = {
    "up": 0, "down": 1, "left": 2, "right": 3,
    "left_up": 4, "left_down": 5, "right_up": 6, "right_down": 7,
}


@app.post("/api/cameras/{cam_id}/ptz")
async def camera_ptz(cam_id: str, req: EzvizPtzRequest):
    """PTZ control — works for both ONVIF and EZVIZ cameras."""
    cam = camera_mgr.get(cam_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    if cam.type == "ezviz":
        token_info = ezviz_mgr.get_token()
        if not token_info:
            raise HTTPException(503, "Failed to get EZVIZ token")

        ptz_code = EZVIZ_PTZ_MAP.get(req.direction)
        if ptz_code is None:
            raise HTTPException(400, f"Unknown direction: {req.direction}")

        import requests as req_lib
        resp = req_lib.post(
            "https://isgpopen.ezvizlife.com/api/lapp/device/ptz/start",
            data={
                "accessToken": token_info["accessToken"],
                "deviceSerial": cam.ezviz_serial,
                "channelNo": cam.ezviz_channel,
                "direction": ptz_code,
                "speed": req.speed,
            },
            timeout=10,
        )
        return resp.json()
    else:
        raise HTTPException(400, "PTZ not supported for this camera type")


@app.post("/api/cameras/{cam_id}/ptz/stop")
async def camera_ptz_stop(cam_id: str):
    """Stop PTZ movement."""
    cam = camera_mgr.get(cam_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    if cam.type == "ezviz":
        token_info = ezviz_mgr.get_token()
        if not token_info:
            raise HTTPException(503, "Failed to get EZVIZ token")

        import requests as req_lib
        resp = req_lib.post(
            "https://isgpopen.ezvizlife.com/api/lapp/device/ptz/stop",
            data={
                "accessToken": token_info["accessToken"],
                "deviceSerial": cam.ezviz_serial,
                "channelNo": cam.ezviz_channel,
            },
            timeout=10,
        )
        return resp.json()
    else:
        raise HTTPException(400, "PTZ not supported for this camera type")


class GateCommand(BaseModel):
    channel: int
    label: str

@app.post("/api/gate/command")
async def gate_command(cmd: GateCommand):
    """Send gate command from UI buttons."""
    if cmd.channel not in (0, 1, 2):
        raise HTTPException(400, "Invalid channel")
    _send_gate_command(cmd.channel, cmd.label)
    return {"ok": True, "channel": cmd.channel, "label": cmd.label}


@app.get("/api/gate/auto/status")
async def gate_auto_status():
    """Get gate automation status."""
    return gate_auto.get_status()


@app.put("/api/cameras/{cam_id}/toggle")
async def toggle_camera(cam_id: str):
    """Enable or disable a camera stream."""
    cam = camera_mgr.get(cam_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    cam.enabled = not cam.enabled
    camera_mgr._save()

    if cam.enabled:
        stream_mgr.start_stream(cam)
        audio_mgr.start_audio(cam)
        face_detector.add_camera(cam.id)
        logger.info(f"Camera {cam.name} enabled")
    else:
        face_detector.remove_camera(cam_id)
        stream_mgr.stop_stream(cam_id)
        audio_mgr.stop_audio(cam_id)
        logger.info(f"Camera {cam.name} disabled")

    return {"ok": True, "id": cam.id, "enabled": cam.enabled}


@app.delete("/api/cameras/{cam_id}")
async def remove_camera(cam_id: str):
    face_detector.remove_camera(cam_id)
    stream_mgr.stop_stream(cam_id)
    if not camera_mgr.remove(cam_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    logger.info(f"Removed camera {cam_id}")
    return {"ok": True}


class QualityRequest(BaseModel):
    quality: str  # "high" or "low"


@app.put("/api/cameras/{cam_id}/quality")
async def switch_quality(cam_id: str, req: QualityRequest):
    cam = camera_mgr.get(cam_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    if cam.type != "rtsp":
        raise HTTPException(400, "Quality switch only for RTSP cameras")

    old_url = cam.url
    # Hikvision: /Streaming/Channels/101 (main) vs /Streaming/Channels/102 (sub)
    if "/Streaming/Channels/" in old_url:
        base = old_url.rsplit("/Streaming/Channels/", 1)[0]
        channel = "101" if req.quality == "high" else "102"
        new_url = f"{base}/Streaming/Channels/{channel}"
    else:
        raise HTTPException(400, "Cannot detect channel pattern in URL")

    if new_url == old_url:
        current = "high" if "101" in old_url else "low"
        return {"ok": True, "quality": current, "changed": False}

    # Stop old stream, update URL, start new stream
    stream_mgr.stop_stream(cam_id)
    cam.url = new_url
    camera_mgr._save()
    stream_mgr.start_stream(cam)

    quality = "high" if "101" in new_url else "low"
    logger.info(f"Switched {cam.name} to {quality} quality: {new_url}")
    return {"ok": True, "quality": quality, "changed": True}


@app.post("/api/discover")
async def discover():
    logger.info("Starting ONVIF discovery...")
    found = discover_onvif_cameras(timeout=5)
    logger.info(f"Discovery found {len(found)} camera(s)")
    results = []
    for cam in found:
        rtsp_url = get_rtsp_url_from_onvif(cam.xaddr)
        results.append({
            "address": cam.address,
            "name": cam.name,
            "xaddr": cam.xaddr,
            "rtsp_url": rtsp_url,
        })
    return results


# --- MJPEG Live Stream ---

@app.get("/streams/{cam_id}/live")
async def mjpeg_feed(cam_id: str):
    """MJPEG stream — multipart/x-mixed-replace for near-real-time display."""
    cam = camera_mgr.get(cam_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    async def generate():
        while True:
            frame = stream_mgr.get_frame(cam_id)
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n"
                    b"\r\n" + frame + b"\r\n"
                )
            await asyncio.sleep(0.05)  # ~20 checks/sec, actual fps limited by FFmpeg

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store", "Access-Control-Allow-Origin": "*"},
    )


@app.get("/streams/{cam_id}/audio")
async def audio_feed(cam_id: str):
    """MP3 audio stream for a camera."""
    cam = camera_mgr.get(cam_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    async def generate():
        while True:
            chunk = audio_mgr.get_chunk(cam_id)
            if chunk:
                yield chunk
            await asyncio.sleep(0.1)

    return StreamingResponse(
        generate(),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-cache, no-store", "Access-Control-Allow-Origin": "*"},
    )


@app.get("/streams/{cam_id}/snapshot")
async def snapshot(cam_id: str):
    """Single JPEG frame — useful for AI detection pipelines."""
    cam = camera_mgr.get(cam_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    frame = stream_mgr.get_frame(cam_id)
    if not frame:
        raise HTTPException(503, "No frame available yet")

    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"},
    )


# --- Face API ---

class LabelFaceRequest(BaseModel):
    name: str


@app.get("/api/faces")
async def list_faces(status: Optional[str] = Query(None)):
    faces = face_mgr.get_faces(status=status)
    result = []
    outfits = face_detector.get_all_outfits()
    for f in faces:
        entry = {
            "id": f.id,
            "name": f.name,
            "image_url": f"/api/faces/{f.id}/image",
            "first_seen": f.first_seen,
            "last_seen": f.last_seen,
            "camera_id": f.camera_id,
            "status": f.status,
            "encoding_count": len(f.encodings) if f.encodings else 0,
        }
        outfit = outfits.get(f.id)
        if outfit:
            entry["shirt_color"] = outfit.get("shirt_color", "")
            entry["shirt_hex"] = outfit.get("shirt_hex", "")
            entry["pants_color"] = outfit.get("pants_color", "")
            entry["pants_hex"] = outfit.get("pants_hex", "")
            # Backward compat
            entry["outfit_color"] = outfit.get("shirt_color", "")
            entry["outfit_hex"] = outfit.get("shirt_hex", "")
        result.append(entry)
    return result


@app.get("/api/faces/{face_id}/image")
async def face_image(face_id: str):
    face = face_mgr.get_face(face_id)
    if not face:
        raise HTTPException(404, "Face not found")
    img_path = Path(face.image_path)
    if not img_path.exists():
        raise HTTPException(404, "Face image not found")
    return Response(
        content=img_path.read_bytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "max-age=3600"},
    )


@app.put("/api/faces/{face_id}")
async def label_face(face_id: str, req: LabelFaceRequest):
    face = face_mgr.label_face(face_id, req.name)
    if not face:
        raise HTTPException(404, "Face not found")
    return {"ok": True, "id": face.id, "name": face.name, "status": face.status}


@app.put("/api/faces/{face_id}/ignore")
async def ignore_face(face_id: str):
    face = face_mgr.ignore_face(face_id)
    if not face:
        raise HTTPException(404, "Face not found")
    return {"ok": True, "id": face.id, "status": face.status}


@app.delete("/api/faces/{face_id}")
async def delete_face(face_id: str):
    if not face_mgr.delete_face(face_id):
        raise HTTPException(404, "Face not found")
    return {"ok": True}


@app.get("/api/faces/{face_id}/outfits")
async def face_outfits(face_id: str):
    """Outfit history for a customer (last 90 days)."""
    history = face_detector.get_outfit_history(face_id)
    return history


@app.get("/api/sightings")
async def list_sightings(
    face_id: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    sightings = face_mgr.get_sightings(face_id=face_id, camera_id=camera_id, limit=limit)
    result = []
    for s in sightings:
        face = face_mgr.get_face(s.face_id)
        result.append({
            "id": s.id,
            "face_id": s.face_id,
            "face_name": face.name if face else "Deleted",
            "face_image_url": f"/api/faces/{s.face_id}/image" if face else "",
            "camera_id": s.camera_id,
            "timestamp": s.timestamp,
            "confidence": s.confidence,
            "image_url": f"/api/sightings/{s.id}/image" if s.image_path else "",
        })
    return result


@app.get("/api/detections")
async def active_detections(cam_id: Optional[str] = Query(None)):
    """Active known-face detections for greeting overlay."""
    return face_detector.get_active_detections(cam_id=cam_id)


@app.get("/api/tracking")
async def tracked_persons(cam_id: Optional[str] = Query(None)):
    """Currently tracked persons with bounding boxes for canvas overlay."""
    return face_detector.get_tracked_persons(cam_id=cam_id)


@app.get("/api/pets")
async def active_pets(cam_id: Optional[str] = Query(None)):
    """Currently detected pets with bounding boxes."""
    return face_detector.get_active_pets(cam_id=cam_id)


@app.get("/api/sightings/{sighting_id}/image")
async def sighting_image(sighting_id: str):
    for s in face_mgr.sightings:
        if s.id == sighting_id:
            if not s.image_path:
                raise HTTPException(404, "No image for this sighting")
            img_path = Path(s.image_path)
            if not img_path.exists():
                raise HTTPException(404, "Sighting image not found")
            return Response(
                content=img_path.read_bytes(),
                media_type="image/jpeg",
                headers={"Cache-Control": "max-age=3600"},
            )
    raise HTTPException(404, "Sighting not found")


# --- Smart Home ---

@app.get("/api/smarthome/devices")
async def smarthome_devices():
    """List discovered smart home devices."""
    return {
        "devices": smart_home.get_devices(),
        "scanning": smart_home.is_scanning,
        "progress": smart_home.scan_progress,
    }


@app.post("/api/smarthome/scan")
async def smarthome_scan():
    """Start a LAN scan for smart home devices."""
    started = smart_home.start_scan()
    return {"started": started, "scanning": smart_home.is_scanning}


@app.post("/api/smarthome/toggle/{device_id:path}")
async def smarthome_toggle(device_id: str):
    """Toggle a switch/plug device."""
    result = smart_home.toggle(device_id)
    if result is None:
        raise HTTPException(404, "Device not found or toggle not supported")
    return result


@app.get("/api/smarthome/status/{device_id:path}")
async def smarthome_status(device_id: str):
    """Refresh and return device status."""
    result = smart_home.refresh_status(device_id)
    if result is None:
        raise HTTPException(404, "Device not found")
    return result


# --- Layout API ---

@app.get("/api/layout")
async def get_layout():
    """Return full layout + detection config."""
    return layout_mgr.get_config()


class SetActiveLayoutRequest(BaseModel):
    layout_id: str

@app.post("/api/layout/active")
async def set_active_layout(req: SetActiveLayoutRequest):
    if not layout_mgr.set_active_layout(req.layout_id):
        raise HTTPException(400, "Invalid layout ID")
    return {"ok": True, "active_layout": req.layout_id}


class UpdateSlotRequest(BaseModel):
    layout_id: str
    slot_type: str  # "slots", "big", or "small"
    index: int
    camera_id: Optional[str] = None
    name: Optional[str] = None

@app.post("/api/layout/slot")
async def update_slot(req: UpdateSlotRequest):
    if not layout_mgr.update_slot(req.layout_id, req.slot_type, req.index,
                                   camera_id=req.camera_id, name=req.name):
        raise HTTPException(400, "Invalid slot or duplicate camera")
    return {"ok": True}


class SwapSlotsRequest(BaseModel):
    layout_id: str
    from_type: str
    from_index: int
    to_type: str
    to_index: int

@app.post("/api/layout/swap")
async def swap_slots(req: SwapSlotsRequest):
    if not layout_mgr.swap_slots(req.layout_id,
                                  req.from_type, req.from_index,
                                  req.to_type, req.to_index):
        raise HTTPException(400, "Invalid swap")
    return {"ok": True}


class MotionModeRequest(BaseModel):
    mode: str  # "auto" or "manual"

@app.post("/api/layout/motion-mode")
async def set_motion_mode(req: MotionModeRequest):
    if not layout_mgr.set_motion_mode(req.mode):
        raise HTTPException(400, "Invalid mode, use 'auto' or 'manual'")
    return {"ok": True, "motion_mode": req.mode}


# --- Detection Settings API ---

class DetectionSettingRequest(BaseModel):
    feature: str        # "motion", "human", "face", "pet"
    enabled: Optional[bool] = None
    sensitivity: Optional[int] = None

@app.post("/api/detection")
async def update_detection(req: DetectionSettingRequest):
    """Toggle a detection feature on/off and/or set sensitivity (0-100)."""
    if not layout_mgr.update_detection(req.feature, enabled=req.enabled,
                                        sensitivity=req.sensitivity):
        raise HTTPException(400, f"Unknown feature: {req.feature}")

    # Apply motion detection state change immediately
    if req.feature == "motion":
        motion_detector.set_enabled(layout_mgr.is_detection_enabled("motion"))
        motion_detector.sensitivity = layout_mgr.get_sensitivity("motion")

    return {"ok": True, "detection": layout_mgr.detection}


@app.get("/api/detection")
async def get_detection():
    """Return current detection settings."""
    return layout_mgr.detection


# --- WebSocket: Motion Events ---

@app.websocket("/ws/motion")
async def motion_ws(websocket: WebSocket):
    await websocket.accept()
    _motion_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _motion_clients:
            _motion_clients.remove(websocket)


# --- Self-Learning API ---

class FeedbackRequest(BaseModel):
    detection_id: str
    correct: bool
    label: str


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Submit user feedback on a detection result."""
    # Try to get crop image from the detection's sighting
    image_data = None
    for s in face_mgr.sightings:
        if s.id == req.detection_id and s.image_path:
            img_path = Path(s.image_path)
            if img_path.exists():
                image_data = img_path.read_bytes()
            break

    feedback = self_learn.submit_feedback(
        detection_id=req.detection_id,
        correct=req.correct,
        label=req.label,
        image_data=image_data,
    )
    return {"ok": True, "feedback_id": feedback.id, "ready_to_retrain": self_learn.should_retrain()}


@app.get("/api/feedback/summary")
async def feedback_summary():
    """Return feedback counts: correct/incorrect per class."""
    return self_learn.feedback_collector.get_summary()


@app.get("/api/model-stats")
async def model_stats():
    """Return model accuracy, version, feedback total, last retrain date."""
    return self_learn.get_model_stats()


@app.post("/api/retrain")
async def trigger_retrain(force: bool = Query(False)):
    """Trigger model retrain. Use force=true to skip minimum data check."""
    import asyncio
    result = await asyncio.to_thread(self_learn.retrain, force=force)
    return result


class RollbackRequest(BaseModel):
    version: str


@app.post("/api/model-rollback")
async def model_rollback(req: RollbackRequest):
    """Rollback to a previous model version."""
    version = self_learn.model_manager.rollback(req.version)
    if not version:
        raise HTTPException(404, f"Version '{req.version}' not found or model file missing")
    return {"ok": True, "active_version": version.version, "accuracy": version.accuracy}


# --- Static Files (Web UI) ---

@app.get("/")
async def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Web UI not found. Place index.html in static/"}


if __name__ == "__main__":
    import socket
    def _get_lan_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "unknown"

    lan_ip = _get_lan_ip()
    print("=" * 50)
    print("  LAN Camera Viewer")
    print(f"  Local:   http://localhost:8080")
    print(f"  Network: http://{lan_ip}:8080")
    print()
    print("  Share the Network URL with others on your LAN")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8080)
