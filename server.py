"""LAN Camera Viewer — Main Server

Start with:  python server.py
Open:         http://localhost:8080
"""
import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from camera_manager import CameraManager, StreamManager
from discovery import discover_onvif_cameras, get_rtsp_url_from_onvif

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

camera_mgr = CameraManager()
stream_mgr = StreamManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: start streams for all saved cameras
    for cam in camera_mgr.list_all():
        if cam.enabled:
            stream_mgr.start_stream(cam)
    logger.info(f"Loaded {len(camera_mgr.cameras)} camera(s)")
    yield
    # Shutdown: stop all streams
    stream_mgr.stop_all()
    logger.info("All streams stopped")


app = FastAPI(title="LAN Camera Viewer", lifespan=lifespan)

STREAMS_DIR = Path("streams")
STATIC_DIR = Path("static")


# --- Models ---

class AddCameraRequest(BaseModel):
    name: str
    url: str
    type: str = "rtsp"  # "rtsp" or "mjpeg"


# --- API Endpoints ---

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
            "stream_url": (
                f"/streams/{cam.id}/stream.m3u8" if cam.type == "rtsp" else cam.url
            ),
        }
        for cam in camera_mgr.list_all()
    ]


@app.post("/api/cameras")
async def add_camera(req: AddCameraRequest):
    cam = camera_mgr.add(req.name, req.url, req.type)
    if cam.type == "rtsp":
        stream_mgr.start_stream(cam)
    logger.info(f"Added camera: {cam.name} ({cam.type}) -> {cam.url}")
    return {"id": cam.id, "name": cam.name}


@app.delete("/api/cameras/{cam_id}")
async def remove_camera(cam_id: str):
    stream_mgr.stop_stream(cam_id)
    if not camera_mgr.remove(cam_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    logger.info(f"Removed camera {cam_id}")
    return {"ok": True}


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


# --- HLS Stream Serving ---

@app.get("/streams/{cam_id}/{filename}")
async def serve_stream(cam_id: str, filename: str):
    file_path = STREAMS_DIR / cam_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Stream not ready")

    if filename.endswith(".m3u8"):
        content_type = "application/vnd.apple.mpegurl"
    else:
        content_type = "video/mp2t"

    return FileResponse(
        file_path,
        media_type=content_type,
        headers={"Cache-Control": "no-cache, no-store", "Access-Control-Allow-Origin": "*"},
    )


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
