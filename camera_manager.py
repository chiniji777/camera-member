import json
import uuid
import subprocess
import shutil
import time
import threading
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

CAMERAS_FILE = Path("cameras.json")
STREAMS_DIR = Path("streams")


@dataclass
class Camera:
    id: str
    name: str
    url: str          # RTSP URL or MJPEG agent URL
    type: str          # "rtsp" or "mjpeg"
    enabled: bool = True


class CameraManager:
    def __init__(self):
        self.cameras: dict[str, Camera] = {}
        self._load()

    def _load(self):
        if CAMERAS_FILE.exists():
            try:
                data = json.loads(CAMERAS_FILE.read_text())
                for item in data:
                    cam = Camera(**item)
                    self.cameras[cam.id] = cam
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed to load cameras.json: {e}")

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
            logger.error("FFmpeg not found in PATH. Install FFmpeg to use RTSP cameras.")
            return

        playlist = stream_dir / "stream.m3u8"

        cmd = [
            ffmpeg_path,
            "-rtsp_transport", "tcp",
            "-i", camera.url,
            "-c:v", "copy",
            "-an",
            "-f", "hls",
            "-hls_time", "3",
            "-hls_list_size", "3",
            "-hls_flags", "delete_segments",
            str(playlist),
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
            self.processes[camera.id] = proc
            self._restart_counts[camera.id] = 0

            t = threading.Thread(target=self._monitor, args=(camera,), daemon=True)
            t.start()

            logger.info(f"Started stream for {camera.name} ({camera.id})")
        except Exception as e:
            logger.error(f"Failed to start FFmpeg for {camera.name}: {e}")

    def _monitor(self, camera: Camera):
        while self._running and camera.id in self.processes:
            proc = self.processes.get(camera.id)
            if proc is None:
                break
            ret = proc.poll()
            if ret is not None and self._running:
                count = self._restart_counts.get(camera.id, 0)
                delay = min(2 ** count, 30)
                logger.warning(f"FFmpeg for {camera.id} exited (code {ret}), restarting in {delay}s")
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
        stream_dir = STREAMS_DIR / cam_id
        if stream_dir.exists():
            shutil.rmtree(stream_dir, ignore_errors=True)

    def stop_all(self):
        self._running = False
        for cam_id in list(self.processes.keys()):
            self.stop_stream(cam_id)
