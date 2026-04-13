import json
import uuid
import subprocess
import shutil
import socket
import time
import threading
import logging
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

CAMERAS_FILE = Path("cameras.json")
STREAMS_DIR = Path("streams")


@dataclass
class Camera:
    id: str
    name: str
    url: str          # RTSP URL, MJPEG agent URL, or ezopen:// URL
    type: str          # "rtsp", "mjpeg", or "ezviz"
    enabled: bool = True
    ezviz_serial: str = ""
    ezviz_channel: int = 1


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


# --- Auto-resolve camera IP on network change ---

def _get_lan_subnet() -> Optional[str]:
    """Get current LAN subnet prefix (e.g. '192.168.1')."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ".".join(ip.split(".")[:3])
    except Exception:
        return None


def _parse_rtsp_url(url: str) -> Optional[dict]:
    """Extract host, port, credentials, and path from an RTSP URL."""
    try:
        parsed = urlparse(url)
        if parsed.scheme != "rtsp":
            return None
        return {
            "username": parsed.username or "",
            "password": parsed.password or "",
            "host": parsed.hostname or "",
            "port": parsed.port or 554,
            "path": parsed.path + ("?" + parsed.query if parsed.query else ""),
        }
    except Exception:
        return None


def _build_rtsp_url(parts: dict, new_host: str) -> str:
    """Rebuild RTSP URL with a new host IP."""
    creds = ""
    if parts["username"]:
        creds = f"{parts['username']}:{parts['password']}@"
    port = f":{parts['port']}" if parts["port"] != 554 else ":554"
    return f"rtsp://{creds}{new_host}{port}{parts['path']}"


def _is_host_reachable(host: str, port: int = 554, timeout: float = 2.0) -> bool:
    """Check if a host has the given port open."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        result = s.connect_ex((host, port))
        s.close()
        return result == 0
    except Exception:
        return False


def _scan_lan_for_rtsp(subnet: str, exclude: str = "", timeout: float = 1.5) -> list[str]:
    """Scan subnet for hosts with port 554 open. Returns list of IPs."""
    def check(ip):
        if ip == exclude:
            return None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            if s.connect_ex((ip, 554)) == 0:
                s.close()
                return ip
            s.close()
        except Exception:
            pass
        return None

    found = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
        futures = {ex.submit(check, f"{subnet}.{i}"): i for i in range(1, 255)}
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            if r:
                found.append(r)
    return sorted(found, key=lambda x: int(x.split(".")[-1]))


def _verify_rtsp(url: str, timeout: float = 8.0) -> bool:
    """Quick check: can FFmpeg open this RTSP URL and get stream info?"""
    ffmpeg_path = _find_ffmpeg()
    if not ffmpeg_path:
        return False
    cmd = [
        ffmpeg_path, "-rtsp_transport", "tcp",
        "-i", url, "-t", "1", "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        # FFmpeg writes stream info to stderr; look for "Video:" as success indicator
        return b"Video:" in proc.stderr
    except Exception:
        return False


def auto_resolve_cameras(camera_mgr: "CameraManager") -> int:
    """Check all RTSP cameras and re-discover any that moved IP.
    Returns number of cameras that were updated."""
    updated = 0
    subnet = _get_lan_subnet()
    if not subnet:
        logger.warning("Cannot determine LAN subnet for auto-resolve")
        return 0

    for cam in camera_mgr.list_all():
        if cam.type != "rtsp":
            continue

        parts = _parse_rtsp_url(cam.url)
        if not parts:
            continue

        # Check if current IP works
        if _is_host_reachable(parts["host"], parts["port"]):
            logger.info(f"Camera {cam.name} reachable at {parts['host']}")
            continue

        logger.warning(f"Camera {cam.name} unreachable at {parts['host']}, scanning LAN...")

        # Scan for candidates
        candidates = _scan_lan_for_rtsp(subnet, exclude=parts["host"])
        if not candidates:
            logger.error(f"No RTSP hosts found on {subnet}.0/24 for {cam.name}")
            continue

        logger.info(f"Found {len(candidates)} RTSP candidate(s): {candidates}")

        # Try each candidate with the saved credentials
        for candidate_ip in candidates:
            new_url = _build_rtsp_url(parts, candidate_ip)
            logger.info(f"Trying {cam.name} at {candidate_ip}...")
            if _verify_rtsp(new_url):
                old_host = parts["host"]
                cam.url = new_url
                camera_mgr._save()
                logger.info(
                    f"Camera {cam.name} relocated: {old_host} -> {candidate_ip}"
                )
                updated += 1
                break
        else:
            logger.error(f"Could not find {cam.name} on any LAN host")

    return updated


def _find_ffmpeg() -> Optional[str]:
    path = shutil.which("ffmpeg")
    if path:
        return path
    # Common install locations
    for fallback in [
        Path("/opt/homebrew/bin/ffmpeg"),
        Path("/usr/local/bin/ffmpeg"),
        Path("C:/ffmpeg/ffmpeg-8.0.1-essentials_build/bin/ffmpeg.exe"),
    ]:
        if fallback.exists():
            return str(fallback)
    return None


class StreamManager:
    """Manages FFmpeg processes that decode RTSP/HLS to MJPEG frames via pipe."""

    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}
        self._restart_counts: dict[str, int] = {}
        self._running = True
        self._frames: dict[str, bytes] = {}       # latest JPEG per camera
        self._frame_locks: dict[str, threading.Lock] = {}
        self._cameras: dict[str, Camera] = {}      # keep ref for restart
        self._camera_mgr: Optional["CameraManager"] = None  # for auto-resolve
        self._ezviz_mgr = None  # set externally for EZVIZ HLS URL refresh
        self.motion_detector = None  # set externally from server.py
        STREAMS_DIR.mkdir(exist_ok=True)

    def set_camera_manager(self, mgr: "CameraManager"):
        """Link to CameraManager so we can auto-resolve IPs on reconnect."""
        self._camera_mgr = mgr

    def set_ezviz_manager(self, mgr):
        """Link to EzvizTokenManager for HLS URL refresh."""
        self._ezviz_mgr = mgr

    def _get_ezviz_hls_url(self, camera: Camera) -> Optional[str]:
        """Get a fresh HLS URL for an EZVIZ camera from the cloud API."""
        if not self._ezviz_mgr:
            logger.error("EzvizTokenManager not set, cannot get HLS URL")
            return None
        token_info = self._ezviz_mgr.get_token()
        if not token_info:
            logger.error("Failed to get EZVIZ token for stream")
            return None
        import requests as req_lib
        try:
            resp = req_lib.post(
                "https://isgpopen.ezvizlife.com/api/lapp/v2/live/address/get",
                data={
                    "accessToken": token_info["accessToken"],
                    "deviceSerial": camera.ezviz_serial or camera.id,
                    "channelNo": camera.ezviz_channel or 1,
                    "protocol": 2,
                    "quality": 1,
                },
                timeout=10,
            )
            data = resp.json()
            if data.get("code") == "200":
                return data["data"]["url"]
            logger.error(f"EZVIZ HLS API error: {data.get('msg', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to get EZVIZ HLS URL: {e}")
        return None

    def start_stream(self, camera: Camera):
        if camera.type not in ("rtsp", "ezviz"):
            return

        ffmpeg_path = _find_ffmpeg()
        if not ffmpeg_path:
            logger.error("FFmpeg not found. Install FFmpeg to use cameras.")
            return

        # For EZVIZ cameras, get HLS URL from cloud API
        input_url = camera.url
        if camera.type == "ezviz":
            hls_url = self._get_ezviz_hls_url(camera)
            if not hls_url:
                logger.error(f"Cannot start EZVIZ stream for {camera.name}: no HLS URL")
                return
            input_url = hls_url

        self._cameras[camera.id] = camera
        self._frame_locks.setdefault(camera.id, threading.Lock())

        stream_dir = STREAMS_DIR / camera.id
        stream_dir.mkdir(exist_ok=True)
        ffmpeg_log = stream_dir / "ffmpeg.log"

        if camera.type == "rtsp":
            cmd = [
                ffmpeg_path,
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-rtsp_transport", "tcp",
                "-i", input_url,
                "-an",
                "-c:v", "mjpeg",
                "-q:v", "5",
                "-f", "image2pipe",
                "-vf", "fps=10",
                "pipe:1",
            ]
        else:
            # EZVIZ HLS input — different flags for HLS
            cmd = [
                ffmpeg_path,
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-i", input_url,
                "-an",
                "-c:v", "mjpeg",
                "-q:v", "5",
                "-f", "image2pipe",
                "-vf", "fps=10",
                "pipe:1",
            ]

        try:
            log_fh = open(ffmpeg_log, "w")
            logger.info(f"FFmpeg cmd: {' '.join(cmd)}")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=log_fh,
                bufsize=0,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
            self.processes[camera.id] = proc
            self._restart_counts[camera.id] = 0

            # Reader thread: parse JPEG frames from pipe
            t = threading.Thread(
                target=self._read_frames, args=(camera.id, proc), daemon=True
            )
            t.start()

            logger.info(f"Started stream for {camera.name} ({camera.id})")
        except Exception as e:
            logger.error(f"Failed to start FFmpeg for {camera.name}: {e}")

    def _read_frames(self, cam_id: str, proc: subprocess.Popen):
        """Read JPEG frames from FFmpeg stdout pipe."""
        buf = b""
        pipe = proc.stdout
        try:
            while self._running and cam_id in self.processes:
                chunk = pipe.read(4096)
                if not chunk:
                    break
                buf += chunk

                # Extract complete JPEG frames (SOI=FFD8, EOI=FFD9)
                while True:
                    start = buf.find(b"\xff\xd8")
                    if start == -1:
                        buf = b""
                        break
                    end = buf.find(b"\xff\xd9", start + 2)
                    if end == -1:
                        # Keep from start marker, wait for more data
                        buf = buf[start:]
                        break
                    # Complete frame
                    frame = buf[start : end + 2]
                    with self._frame_locks[cam_id]:
                        self._frames[cam_id] = frame
                    # Feed frame to motion detector
                    if self.motion_detector:
                        try:
                            self.motion_detector.check_frame(cam_id, frame)
                        except Exception:
                            pass
                    buf = buf[end + 2 :]
        except Exception as e:
            logger.error(f"Frame reader error for {cam_id}: {e}")
        finally:
            # FFmpeg exited — restart if still running
            if self._running and cam_id in self.processes:
                count = self._restart_counts.get(cam_id, 0)
                delay = min(2 ** count, 30)
                logger.warning(
                    f"FFmpeg for {cam_id} exited (code {proc.poll()}), restarting in {delay}s"
                )
                time.sleep(delay)
                self._restart_counts[cam_id] = count + 1

                # After 3 consecutive failures, try auto-resolving the IP
                if count >= 2 and self._camera_mgr and cam_id in self._cameras:
                    cam = self._cameras[cam_id]
                    parts = _parse_rtsp_url(cam.url)
                    if parts and not _is_host_reachable(parts["host"], parts["port"]):
                        logger.info(f"Auto-resolving IP for {cam.name}...")
                        auto_resolve_cameras(self._camera_mgr)
                        # Reload camera ref (URL may have changed)
                        refreshed = self._camera_mgr.get(cam_id)
                        if refreshed:
                            self._cameras[cam_id] = refreshed

                if self._running and cam_id in self._cameras:
                    self.processes.pop(cam_id, None)
                    self.start_stream(self._cameras[cam_id])

    def get_frame(self, cam_id: str) -> Optional[bytes]:
        """Get the latest JPEG frame for a camera."""
        lock = self._frame_locks.get(cam_id)
        if lock is None:
            return None
        with lock:
            return self._frames.get(cam_id)

    def stop_stream(self, cam_id: str):
        proc = self.processes.pop(cam_id, None)
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._frames.pop(cam_id, None)
        self._cameras.pop(cam_id, None)
        stream_dir = STREAMS_DIR / cam_id
        if stream_dir.exists():
            shutil.rmtree(stream_dir, ignore_errors=True)

    def stop_all(self):
        self._running = False
        for cam_id in list(self.processes.keys()):
            self.stop_stream(cam_id)


class AudioStreamManager:
    """Manages FFmpeg processes that extract audio from RTSP as MP3."""

    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}
        self._running = True
        self._chunks: dict[str, bytes] = {}
        self._chunk_locks: dict[str, threading.Lock] = {}
        self._cameras: dict[str, Camera] = {}

    def start_audio(self, camera: Camera):
        if camera.type != "rtsp":
            return

        ffmpeg_path = _find_ffmpeg()
        if not ffmpeg_path:
            return

        self._cameras[camera.id] = camera
        self._chunk_locks.setdefault(camera.id, threading.Lock())

        cmd = [
            ffmpeg_path,
            "-rtsp_transport", "tcp",
            "-i", camera.url,
            "-vn",
            "-acodec", "libmp3lame",
            "-ab", "64k",
            "-ar", "8000",
            "-ac", "1",
            "-f", "mp3",
            "pipe:1",
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
            self.processes[camera.id] = proc

            t = threading.Thread(
                target=self._read_audio, args=(camera.id, proc), daemon=True
            )
            t.start()
            logger.info(f"Started audio stream for {camera.name} ({camera.id})")
        except Exception as e:
            logger.error(f"Failed to start audio FFmpeg for {camera.name}: {e}")

    def _read_audio(self, cam_id: str, proc: subprocess.Popen):
        pipe = proc.stdout
        try:
            while self._running and cam_id in self.processes:
                chunk = pipe.read(4096)
                if not chunk:
                    break
                with self._chunk_locks[cam_id]:
                    self._chunks[cam_id] = chunk
        except Exception as e:
            logger.error(f"Audio reader error for {cam_id}: {e}")
        finally:
            if self._running and cam_id in self._cameras:
                logger.warning(f"Audio FFmpeg for {cam_id} exited, restarting in 2s")
                time.sleep(2)
                self.processes.pop(cam_id, None)
                if self._running and cam_id in self._cameras:
                    self.start_audio(self._cameras[cam_id])

    def get_chunk(self, cam_id: str) -> Optional[bytes]:
        lock = self._chunk_locks.get(cam_id)
        if lock is None:
            return None
        with lock:
            return self._chunks.get(cam_id)

    def stop_audio(self, cam_id: str):
        proc = self.processes.pop(cam_id, None)
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._chunks.pop(cam_id, None)
        self._cameras.pop(cam_id, None)

    def stop_all(self):
        self._running = False
        for cam_id in list(self.processes.keys()):
            self.stop_audio(cam_id)
