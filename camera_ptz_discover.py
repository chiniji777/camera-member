"""PTZ Control + Enhanced Discovery — Add to camera-member server.py"""
import re
import asyncio
import logging
import socket
import concurrent.futures
from pydantic import BaseModel
from fastapi import HTTPException

logger = logging.getLogger(__name__)

HIKVISION_USER = "admin"
HIKVISION_PASS = "Nutsniper0977"


class PTZRequest(BaseModel):
    direction: str  # left, right, up, down, stop, left_up, right_up, left_down, right_down
    speed: int = 50


class PTZPresetRequest(BaseModel):
    preset: int


def get_camera_host(cam_url: str) -> str:
    m = re.search(r"@([\d.]+)", cam_url)
    return m.group(1) if m else ""


def check_ptz_support(host: str) -> bool:
    import requests
    from requests.auth import HTTPDigestAuth
    try:
        r = requests.get(
            f"http://{host}/ISAPI/PTZCtrl/channels/1/capabilities",
            auth=HTTPDigestAuth(HIKVISION_USER, HIKVISION_PASS),
            timeout=3,
        )
        return r.status_code == 200 and b"notSupport" not in r.content
    except Exception:
        return False


def get_camera_model(host: str) -> dict:
    import requests
    from requests.auth import HTTPDigestAuth
    try:
        r = requests.get(
            f"http://{host}/ISAPI/System/deviceInfo",
            auth=HTTPDigestAuth(HIKVISION_USER, HIKVISION_PASS),
            timeout=5,
        )
        if r.status_code == 200:
            model = re.search(r"<model>(.*?)</model>", r.text)
            mac = re.search(r"<macAddress>(.*?)</macAddress>", r.text)
            name = re.search(r"<deviceName>(.*?)</deviceName>", r.text)
            return {
                "model": model.group(1) if model else "",
                "mac": mac.group(1) if mac else "",
                "device_name": name.group(1) if name else "",
            }
    except Exception:
        pass
    return {}


def scan_lan_rtsp(subnet: str, timeout: float = 1.5) -> list:
    def check(ip):
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


def register_ptz_routes(app, camera_mgr, stream_mgr, audio_mgr, face_detector):
    """Register PTZ and discovery routes on the FastAPI app."""

    @app.post("/api/cameras/{cam_id}/ptz")
    async def ptz_control(cam_id: str, req: PTZRequest):
        import requests
        from requests.auth import HTTPDigestAuth

        cam = camera_mgr.get(cam_id)
        if not cam:
            raise HTTPException(404, "Camera not found")

        host = get_camera_host(cam.url)
        if not host:
            raise HTTPException(400, "Cannot extract host from camera URL")

        direction_map = {
            "left":       {"pan": -req.speed, "tilt": 0},
            "right":      {"pan": req.speed,  "tilt": 0},
            "up":         {"pan": 0, "tilt": req.speed},
            "down":       {"pan": 0, "tilt": -req.speed},
            "stop":       {"pan": 0, "tilt": 0},
            "left_up":    {"pan": -req.speed, "tilt": req.speed},
            "right_up":   {"pan": req.speed,  "tilt": req.speed},
            "left_down":  {"pan": -req.speed, "tilt": -req.speed},
            "right_down": {"pan": req.speed,  "tilt": -req.speed},
        }

        if req.direction not in direction_map:
            raise HTTPException(400, f"Invalid direction: {req.direction}")

        vals = direction_map[req.direction]
        xml = f'<PTZData><pan>{vals["pan"]}</pan><tilt>{vals["tilt"]}</tilt><zoom>0</zoom></PTZData>'

        try:
            r = requests.put(
                f"http://{host}/ISAPI/PTZCtrl/channels/1/continuous",
                data=xml,
                headers={"Content-Type": "application/xml"},
                auth=HTTPDigestAuth(HIKVISION_USER, HIKVISION_PASS),
                timeout=5,
            )
            success = r.status_code == 200
            logger.info(f"PTZ {req.direction} on {cam.name}: {r.status_code}")
            return {"ok": success, "direction": req.direction, "status": r.status_code}
        except Exception as e:
            raise HTTPException(500, f"PTZ command failed: {e}")

    @app.post("/api/cameras/{cam_id}/ptz/stop")
    async def ptz_stop(cam_id: str):
        import requests
        from requests.auth import HTTPDigestAuth

        cam = camera_mgr.get(cam_id)
        if not cam:
            raise HTTPException(404, "Camera not found")

        host = get_camera_host(cam.url)
        xml = '<PTZData><pan>0</pan><tilt>0</tilt><zoom>0</zoom></PTZData>'

        try:
            r = requests.put(
                f"http://{host}/ISAPI/PTZCtrl/channels/1/continuous",
                data=xml,
                headers={"Content-Type": "application/xml"},
                auth=HTTPDigestAuth(HIKVISION_USER, HIKVISION_PASS),
                timeout=5,
            )
            return {"ok": r.status_code == 200}
        except Exception as e:
            raise HTTPException(500, f"PTZ stop failed: {e}")

    @app.post("/api/cameras/{cam_id}/ptz/preset")
    async def ptz_goto_preset(cam_id: str, req: PTZPresetRequest):
        import requests
        from requests.auth import HTTPDigestAuth

        cam = camera_mgr.get(cam_id)
        if not cam:
            raise HTTPException(404, "Camera not found")

        host = get_camera_host(cam.url)
        xml = f'<PTZPreset><id>{req.preset}</id></PTZPreset>'

        try:
            r = requests.put(
                f"http://{host}/ISAPI/PTZCtrl/channels/1/presets/{req.preset}/goto",
                data=xml,
                headers={"Content-Type": "application/xml"},
                auth=HTTPDigestAuth(HIKVISION_USER, HIKVISION_PASS),
                timeout=5,
            )
            return {"ok": r.status_code == 200, "preset": req.preset}
        except Exception as e:
            raise HTTPException(500, f"PTZ preset failed: {e}")

    @app.post("/api/discover-and-add")
    async def discover_and_add():
        """Scan LAN for Hikvision cameras, check credentials, auto-add new ones."""
        import requests
        from requests.auth import HTTPDigestAuth

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            subnet = ".".join(ip.split(".")[:3])
        except Exception:
            raise HTTPException(500, "Cannot determine LAN subnet")

        logger.info(f"Scanning {subnet}.0/24 for cameras...")
        candidates = await asyncio.to_thread(scan_lan_rtsp, subnet)
        logger.info(f"Found {len(candidates)} RTSP host(s): {candidates}")

        existing_hosts = set()
        for cam in camera_mgr.list_all():
            m = re.search(r"@([\d.]+)", cam.url)
            if m:
                existing_hosts.add(m.group(1))

        results = []
        for candidate_ip in candidates:
            info = {"ip": candidate_ip, "added": False, "exists": candidate_ip in existing_hosts}

            try:
                r = requests.get(
                    f"http://{candidate_ip}/ISAPI/System/deviceInfo",
                    auth=HTTPDigestAuth(HIKVISION_USER, HIKVISION_PASS),
                    timeout=5,
                )
                if r.status_code == 200:
                    model = re.search(r"<model>(.*?)</model>", r.text)
                    name = re.search(r"<deviceName>(.*?)</deviceName>", r.text)
                    mac = re.search(r"<macAddress>(.*?)</macAddress>", r.text)
                    info["model"] = model.group(1) if model else "Unknown"
                    info["name"] = name.group(1) if name else f"Camera at {candidate_ip}"
                    info["mac"] = mac.group(1) if mac else ""
                    info["ptz"] = await asyncio.to_thread(check_ptz_support, candidate_ip)

                    if candidate_ip not in existing_hosts:
                        cam_name = info.get("model", f"Camera {candidate_ip}")
                        rtsp_url = f"rtsp://{HIKVISION_USER}:{HIKVISION_PASS}@{candidate_ip}:554/Streaming/Channels/102"
                        cam = camera_mgr.add(cam_name, rtsp_url, "rtsp")
                        stream_mgr.start_stream(cam)
                        audio_mgr.start_audio(cam)
                        face_detector.add_camera(cam.id)
                        info["added"] = True
                        info["cam_id"] = cam.id
                        logger.info(f"Auto-added camera: {cam_name} at {candidate_ip}")
                else:
                    info["error"] = f"Auth failed (HTTP {r.status_code})"
            except Exception as e:
                info["error"] = str(e)

            results.append(info)

        return {"scanned": f"{subnet}.0/24", "found": len(candidates), "cameras": results}

    @app.get("/api/cameras/detailed")
    async def list_cameras_detailed():
        result = []
        for cam in camera_mgr.list_all():
            info = {
                "id": cam.id,
                "name": cam.name,
                "url": cam.url,
                "type": cam.type,
                "enabled": cam.enabled,
                "stream_url": f"/streams/{cam.id}/live" if cam.type == "rtsp" else cam.url,
            }
            m = re.search(r"@([\d.]+)", cam.url)
            if m:
                host = m.group(1)
                info["host"] = host
                info["ptz"] = await asyncio.to_thread(check_ptz_support, host)
                model_info = await asyncio.to_thread(get_camera_model, host)
                info.update(model_info)
            if cam.type == "rtsp" and "/Streaming/Channels/" in cam.url:
                info["quality"] = "high" if "/Channels/101" in cam.url else "low"
                info["can_switch_quality"] = True
            result.append(info)
        return result
