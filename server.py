"""LAN Camera Viewer — Main Server

Start with:  python server.py
Open:         http://localhost:8080
"""
import asyncio
import uvicorn
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, Response
from pydantic import BaseModel
from pathlib import Path
from camera_manager import CameraManager, StreamManager
from discovery import discover_onvif_cameras, get_rtsp_url_from_onvif
from face_manager import FaceManager, FaceDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

camera_mgr = CameraManager()
stream_mgr = StreamManager()
face_mgr = FaceManager()
face_detector = FaceDetector(stream_mgr, face_mgr)


@asynccontextmanager
async def lifespan(app: FastAPI):
    for cam in camera_mgr.list_all():
        if cam.enabled:
            stream_mgr.start_stream(cam)
    logger.info(f"Loaded {len(camera_mgr.cameras)} camera(s)")

    # Start face detection on all active cameras
    active_ids = [cam.id for cam in camera_mgr.list_all() if cam.enabled]
    if active_ids:
        face_detector.start(active_ids)
        logger.info(f"Face detection started on {len(active_ids)} camera(s)")

    yield

    face_detector.stop()
    stream_mgr.stop_all()
    logger.info("All streams stopped")


app = FastAPI(title="LAN Camera Viewer", lifespan=lifespan)

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
            "stream_url": (
                f"/streams/{cam.id}/live" if cam.type == "rtsp" else cam.url
            ),
        }
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
        face_detector.add_camera(cam.id)
    logger.info(f"Added camera: {cam.name} ({cam.type}) -> {cam.url}")
    return {"id": cam.id, "name": cam.name}


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
            entry["outfit_color"] = outfit["color_name"]
            entry["outfit_hex"] = outfit["color_hex"]
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


# --- Static Files (Web UI) ---

@app.get("/")
async def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Web UI not found. Place index.html in static/"}


if __name__ == "__main__":
    print("=" * 50)
    print("  LAN Camera Viewer")
    print("  Open http://localhost:8080 in your browser")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8080)
