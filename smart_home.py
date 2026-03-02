"""Smart Home Device Scanner — Discover WiFi switches & sensors on LAN.

Supports: Tasmota, Shelly, Sonoff (eWeLink), ESPHome, TP-Link Kasa, MQTT brokers.
"""
import socket
import json
import logging
import struct
import time
import threading
from dataclasses import dataclass, field
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("smart_home")

# Ports commonly used by smart home devices
SCAN_PORTS = [80, 8080, 8081, 9999, 1883]


@dataclass
class SmartDevice:
    ip: str
    port: int
    device_type: str   # switch, sensor, light, plug, broker, unknown
    brand: str         # tasmota, shelly, esphome, kasa, mqtt, unknown
    name: str = ""
    mac: str = ""
    model: str = ""
    firmware: str = ""
    status: dict = field(default_factory=dict)
    online: bool = True
    last_seen: float = 0


class SmartHomeScanner:
    def __init__(self):
        self._devices: dict[str, SmartDevice] = {}  # "ip:port" -> device
        self._scanning = False
        self._scan_progress = 0  # 0-100
        self._last_scan = 0.0
        self._lock = threading.Lock()

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_lan_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(1)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "192.168.1.1"

    @staticmethod
    def _get_subnet(ip: str) -> str:
        return ".".join(ip.split(".")[:3])

    @staticmethod
    def _port_open(ip: str, port: int, timeout: float = 0.3) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            ok = s.connect_ex((ip, port)) == 0
            s.close()
            return ok
        except Exception:
            return False

    @staticmethod
    def _http_get(url: str, timeout: float = 2.0) -> Optional[bytes]:
        try:
            req = Request(url, headers={"User-Agent": "LanScanner/1.0"})
            return urlopen(req, timeout=timeout).read()
        except Exception:
            return None

    # ── mDNS discovery ─────────────────────────────────────────────

    def _mdns_query(self, service: str, timeout: float = 3.0) -> list[tuple[str, bytes]]:
        """Send mDNS query and collect responses."""
        MCAST = "224.0.0.251"
        # Build query packet
        query = b"\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00"
        for part in service.split("."):
            query += bytes([len(part)]) + part.encode()
        query += b"\x00\x00\x0c\x00\x01"  # PTR, IN

        results = []
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            sock.settimeout(timeout)
            sock.sendto(query, (MCAST, 5353))
            try:
                while True:
                    data, addr = sock.recvfrom(4096)
                    results.append((addr[0], data))
            except socket.timeout:
                pass
            sock.close()
        except Exception:
            pass
        return results

    def _discover_mdns(self) -> dict[str, SmartDevice]:
        """Discover devices via mDNS service announcements."""
        devices: dict[str, SmartDevice] = {}
        own_ip = self._get_lan_ip()

        # Query for various smart home services
        services = {
            "_ewelink._tcp.local": "sonoff",
            "_esphomelib._tcp.local": "esphome",
            "_hap._tcp.local": "homekit",
            "_http._tcp.local": None,  # generic HTTP
        }

        for service, brand in services.items():
            for ip, data in self._mdns_query(service, timeout=1.5):
                if ip == own_ip:
                    continue
                key = f"{ip}:mdns"
                if key in devices:
                    continue
                text = data.decode("ascii", errors="replace").lower()
                # Extract readable name from mDNS
                name = ""
                if brand == "sonoff" or "ewelink" in text:
                    name = f"Sonoff ({ip})"
                    devices[key] = SmartDevice(
                        ip=ip, port=8081, device_type="switch", brand="sonoff",
                        name=name, model="eWeLink",
                        status={}, last_seen=time.time(),
                    )
                elif brand == "esphome" or "esphome" in text:
                    name = f"ESPHome ({ip})"
                    devices[key] = SmartDevice(
                        ip=ip, port=80, device_type="sensor", brand="esphome",
                        name=name, model="ESPHome",
                        status={}, last_seen=time.time(),
                    )
                elif brand == "homekit" or "_hap" in text:
                    name = f"HomeKit ({ip})"
                    devices[key] = SmartDevice(
                        ip=ip, port=80, device_type="switch", brand="homekit",
                        name=name, model="HomeKit",
                        status={}, last_seen=time.time(),
                    )

        return devices

    # ── device identification ────────────────────────────────────────

    def _try_tasmota(self, ip: str, port: int) -> Optional[SmartDevice]:
        raw = self._http_get(f"http://{ip}:{port}/cm?cmnd=Status%200")
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except Exception:
            return None
        if "Status" not in data and "StatusFWR" not in data:
            return None
        st = data.get("Status", {})
        net = data.get("StatusNET", {})
        fwr = data.get("StatusFWR", {})
        friendly = st.get("FriendlyName", ["Tasmota"])
        name = friendly[0] if isinstance(friendly, list) else str(friendly)
        power = st.get("Power", 0)
        # Determine type from module
        module = st.get("Module", 0)
        dtype = "switch"
        # Modules 0-19 are various relays/sonoff, 25-27 are lights
        if isinstance(module, int) and 25 <= module <= 70:
            dtype = "light"
        return SmartDevice(
            ip=ip, port=port, device_type=dtype, brand="tasmota",
            name=name or f"Tasmota-{ip}",
            mac=net.get("Mac", ""), model=st.get("DeviceName", "Tasmota"),
            firmware=fwr.get("Version", ""),
            status={"power": bool(power)}, last_seen=time.time(),
        )

    def _try_shelly(self, ip: str, port: int) -> Optional[SmartDevice]:
        raw = self._http_get(f"http://{ip}:{port}/shelly")
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except Exception:
            return None
        if "type" not in data and "mac" not in data and "model" not in data:
            return None
        dev_type = data.get("type", data.get("model", "unknown"))
        mac = data.get("mac", "")
        fw = data.get("fw", "")
        # classify
        dtype = "switch"
        lower = dev_type.lower()
        if any(k in lower for k in ("ht", "sensor", "th", "motion", "flood")):
            dtype = "sensor"
        elif any(k in lower for k in ("bulb", "rgbw", "dimmer", "duo", "vintage")):
            dtype = "light"
        elif any(k in lower for k in ("plug", "pm")):
            dtype = "plug"
        # get status
        status = {}
        raw2 = self._http_get(f"http://{ip}:{port}/status")
        if raw2:
            try:
                st = json.loads(raw2)
                relays = st.get("relays", [])
                if relays:
                    status["power"] = relays[0].get("ison", False)
                if "temperature" in st:
                    status["temperature"] = st["temperature"]
                if "humidity" in st:
                    status["humidity"] = st["humidity"]
                # Shelly Gen2 uses different paths but /shelly works for discovery
            except Exception:
                pass
        return SmartDevice(
            ip=ip, port=port, device_type=dtype, brand="shelly",
            name=data.get("name", "") or f"Shelly-{dev_type}",
            mac=mac, model=dev_type, firmware=fw,
            status=status, last_seen=time.time(),
        )

    def _try_esphome(self, ip: str, port: int) -> Optional[SmartDevice]:
        raw = self._http_get(f"http://{ip}:{port}/")
        if not raw:
            return None
        html = raw.decode("utf-8", errors="ignore").lower()
        if "esphome" not in html:
            return None
        # Try to get sensor data from /sensor or /text_sensor
        name = f"ESPHome-{ip}"
        status = {}
        for line in html.split("\n"):
            if "<title>" in line and "</title>" in line:
                t = line.split("<title>")[1].split("</title>")[0].strip()
                if t and t != "ESPHome Web":
                    name = t
                break
        return SmartDevice(
            ip=ip, port=port, device_type="sensor", brand="esphome",
            name=name, model="ESPHome",
            status=status, last_seen=time.time(),
        )

    def _try_sonoff(self, ip: str, port: int) -> Optional[SmartDevice]:
        """Detect Sonoff devices — eWeLink DIY mode (port 8081) or web UI (port 80)."""
        # Method 1: eWeLink DIY mode REST API (typically port 8081)
        info = self._sonoff_diy_request(ip, port, "info")
        if info and "data" in info:
            data = info["data"]
            device_id = data.get("deviceid", "")
            switch_state = data.get("switch", "off")
            name = f"Sonoff-{device_id[-4:]}" if device_id else f"Sonoff-{ip}"
            fw = data.get("fwVersion", "")
            signal = data.get("signalStrength", "")
            status = {"power": switch_state == "on"}
            if signal:
                status["rssi"] = signal
            return SmartDevice(
                ip=ip, port=port, device_type="switch", brand="sonoff",
                name=name, model=data.get("type", "Sonoff"),
                firmware=fw, status=status, last_seen=time.time(),
            )

        # Method 2: Check HTTP page for Sonoff/eWeLink indicators
        raw = self._http_get(f"http://{ip}:{port}/")
        if raw:
            html = raw.decode("utf-8", errors="ignore").lower()
            if any(kw in html for kw in ("sonoff", "ewelink", "coolkit", "itead")):
                # Try /zeroconf/info on same port as fallback
                name = f"Sonoff-{ip}"
                for line in html.split("\n"):
                    if "<title>" in line and "</title>" in line:
                        t = line.split("<title>")[1].split("</title>")[0].strip()
                        if t:
                            name = t
                        break
                return SmartDevice(
                    ip=ip, port=port, device_type="switch", brand="sonoff",
                    name=name, model="Sonoff",
                    status={}, last_seen=time.time(),
                )

        # Method 3: Try DIY mode on port 8081 even if we're checking port 80
        if port == 80:
            info = self._sonoff_diy_request(ip, 8081, "info")
            if info and "data" in info:
                data = info["data"]
                device_id = data.get("deviceid", "")
                switch_state = data.get("switch", "off")
                name = f"Sonoff-{device_id[-4:]}" if device_id else f"Sonoff-{ip}"
                return SmartDevice(
                    ip=ip, port=8081, device_type="switch", brand="sonoff",
                    name=name, model=data.get("type", "Sonoff"),
                    firmware=data.get("fwVersion", ""),
                    status={"power": switch_state == "on"},
                    last_seen=time.time(),
                )

        return None

    @staticmethod
    def _sonoff_diy_request(ip: str, port: int, endpoint: str, data: dict = None) -> Optional[dict]:
        """Send request to Sonoff eWeLink DIY mode API."""
        url = f"http://{ip}:{port}/zeroconf/{endpoint}"
        body = json.dumps({"deviceid": "", "data": data or {}}).encode()
        try:
            req = Request(url, data=body, headers={
                "Content-Type": "application/json",
                "User-Agent": "LanScanner/1.0",
            })
            resp = urlopen(req, timeout=3)
            return json.loads(resp.read())
        except Exception:
            return None

    def _try_generic_http(self, ip: str, port: int) -> Optional[SmartDevice]:
        """Detect generic smart devices by HTTP response."""
        raw = self._http_get(f"http://{ip}:{port}/")
        if not raw:
            return None
        html = raw.decode("utf-8", errors="ignore")[:3000].lower()

        # Skip obvious non-smart devices (routers, NAS, cameras, etc.)
        skip_kw = ["camera", "hikvision", "dahua", "router", "mikrotik", "openwrt",
                    "synology", "qnap", "proxmox", "unifi", "pfsense", "asus",
                    "netgear", "tp-link", "linksys", "d-link", "tenda", "zyxel",
                    "huawei", "cisco", "login", "password", "username"]
        if any(k in html for k in skip_kw):
            return None
        # Skip if the gateway IP (likely a router)
        subnet = self._get_subnet(self._get_lan_ip())
        if ip == f"{subnet}.1":
            return None

        # Check for Arduino/ESP devices (JSON API with device info)
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                # ESP32/Arduino device with JSON API
                is_esp = any(k in data for k in ("free_heap", "flash_size", "uptime",
                                                   "wifi_ip", "mac", "uuid"))
                if is_esp:
                    mac = data.get("mac", "")
                    name = data.get("hostname", data.get("name", f"ESP-{ip}"))
                    model = "ESP32" if data.get("flash_size", 0) > 4000000 else "ESP8266"
                    status = {}
                    if "uptime" in data:
                        status["uptime"] = data["uptime"]
                    return SmartDevice(
                        ip=ip, port=port, device_type="sensor", brand="esp",
                        name=name, mac=mac, model=model,
                        status=status, last_seen=time.time(),
                    )
        except (json.JSONDecodeError, Exception):
            pass

        # Check for smart home indicators in HTML
        smart_kw = ["relay", "smart", "plug", "sensor", "temperature",
                     "humidity", "watt", "kwh", "gpio", "mqtt", "zigbee",
                     "tuya", "sonoff", "blitzwolf", "ewelink"]
        if any(k in html for k in smart_kw):
            dtype = "unknown"
            if any(k in html for k in ("relay", "plug")):
                dtype = "switch"
            elif any(k in html for k in ("sensor", "temperature", "humidity")):
                dtype = "sensor"
            return SmartDevice(
                ip=ip, port=port, device_type=dtype, brand="unknown",
                name=f"Device-{ip}:{port}", model="HTTP Device",
                status={}, last_seen=time.time(),
            )
        return None

    def _try_mqtt(self, ip: str) -> Optional[SmartDevice]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect((ip, 1883))
            # MQTT CONNECT packet (minimal, protocol 3.1.1)
            connect = (
                b"\x10\x10\x00\x04MQTT\x04\x02\x00\x3c"
                b"\x00\x04scan"
            )
            s.send(connect)
            resp = s.recv(4)
            s.close()
            if len(resp) >= 2 and resp[0] == 0x20:
                return SmartDevice(
                    ip=ip, port=1883, device_type="broker", brand="mqtt",
                    name=f"MQTT Broker ({ip})", model="MQTT",
                    status={"connected": True}, last_seen=time.time(),
                )
        except Exception:
            pass
        return None

    def _try_kasa(self, ip: str) -> Optional[SmartDevice]:
        """TP-Link Kasa devices use port 9999 with XOR-encrypted JSON."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect((ip, 9999))
            # Kasa protocol: 4-byte length + XOR-encrypted JSON
            cmd = '{"system":{"get_sysinfo":{}}}'
            key = 171
            enc = bytearray(4)  # length placeholder
            for c in cmd:
                b = key ^ ord(c)
                key = b
                enc.append(b)
            import struct
            enc[:4] = struct.pack(">I", len(cmd))
            s.send(bytes(enc))
            resp = s.recv(4096)
            s.close()
            if len(resp) <= 4:
                return None
            # Decrypt
            key = 171
            dec = []
            for b in resp[4:]:
                dec.append(chr(key ^ b))
                key = b
            data = json.loads("".join(dec))
            info = data.get("system", {}).get("get_sysinfo", {})
            if not info:
                return None
            alias = info.get("alias", f"TP-Link-{ip}")
            model = info.get("model", "Kasa")
            mac = info.get("mac", info.get("mic_mac", ""))
            fw = info.get("sw_ver", "")
            relay = info.get("relay_state", None)
            dtype = "plug" if "plug" in model.lower() else "switch"
            if "bulb" in model.lower() or "light" in model.lower():
                dtype = "light"
            status = {}
            if relay is not None:
                status["power"] = bool(relay)
            return SmartDevice(
                ip=ip, port=9999, device_type=dtype, brand="kasa",
                name=alias, mac=mac, model=model, firmware=fw,
                status=status, last_seen=time.time(),
            )
        except Exception:
            return None

    # ── identification pipeline ──────────────────────────────────────

    def _identify(self, ip: str, port: int) -> Optional[SmartDevice]:
        if port == 1883:
            return self._try_mqtt(ip)
        if port == 9999:
            return self._try_kasa(ip)
        # HTTP ports — try specific brands first, then generic
        for fn in (self._try_tasmota, self._try_shelly, self._try_sonoff,
                   self._try_esphome, self._try_generic_http):
            dev = fn(ip, port)
            if dev:
                return dev
        return None

    # ── scanning ─────────────────────────────────────────────────────

    def scan_network(self):
        if self._scanning:
            return
        self._scanning = True
        self._scan_progress = 0
        new_devices: dict[str, SmartDevice] = {}
        try:
            own_ip = self._get_lan_ip()
            subnet = self._get_subnet(own_ip)
            logger.info(f"Smart home scan: {subnet}.0/24 ...")

            # Phase 0: mDNS discovery (fast, finds devices that hide from port scans)
            logger.info("Phase 0: mDNS discovery...")
            mdns_devices = self._discover_mdns()
            for key, dev in mdns_devices.items():
                new_devices[key] = dev
                logger.info(f"  mDNS: {dev.brand} {dev.device_type}: {dev.name} @ {dev.ip}")
            self._scan_progress = 10

            # Phase 1: fast port scan (parallel)
            open_hosts: dict[str, list[int]] = {}

            def _check(suffix: int):
                ip = f"{subnet}.{suffix}"
                if ip == own_ip:
                    return None
                ports = [p for p in SCAN_PORTS if self._port_open(ip, p)]
                return (ip, ports) if ports else None

            with ThreadPoolExecutor(max_workers=128) as pool:
                for i, result in enumerate(pool.map(_check, range(1, 255))):
                    self._scan_progress = 10 + int((i / 254) * 50)  # 10-60%
                    if result:
                        ip, ports = result
                        open_hosts[ip] = ports

            logger.info(f"Port scan done — {len(open_hosts)} hosts with open ports")

            # Phase 2: identify devices
            tasks = []
            for ip, ports in open_hosts.items():
                for port in ports:
                    tasks.append((ip, port))

            total = max(len(tasks), 1)
            with ThreadPoolExecutor(max_workers=16) as pool:
                futures = {pool.submit(self._identify, ip, port): (ip, port)
                           for ip, port in tasks}
                for i, future in enumerate(futures):
                    self._scan_progress = 60 + int((i / total) * 40)  # 60-100%
                    try:
                        dev = future.result(timeout=10)
                    except Exception:
                        dev = None
                    if dev:
                        key = f"{dev.ip}:{dev.port}"
                        # Don't overwrite mDNS-discovered device with less info
                        if key not in new_devices:
                            new_devices[key] = dev
                            logger.info(f"  Found {dev.brand} {dev.device_type}: {dev.name} @ {dev.ip}:{dev.port}")

            with self._lock:
                self._devices = new_devices
            self._last_scan = time.time()
            self._scan_progress = 100
            logger.info(f"Scan complete — {len(new_devices)} smart home device(s)")
        except Exception as e:
            logger.error(f"Scan error: {e}")
        finally:
            self._scanning = False

    def start_scan(self) -> bool:
        if self._scanning:
            return False
        threading.Thread(target=self.scan_network, daemon=True).start()
        return True

    # ── getters ──────────────────────────────────────────────────────

    @property
    def is_scanning(self) -> bool:
        return self._scanning

    @property
    def scan_progress(self) -> int:
        return self._scan_progress

    def get_devices(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "id": key,
                    "ip": d.ip,
                    "port": d.port,
                    "type": d.device_type,
                    "brand": d.brand,
                    "name": d.name,
                    "mac": d.mac,
                    "model": d.model,
                    "firmware": d.firmware,
                    "status": d.status,
                    "online": d.online,
                }
                for key, d in self._devices.items()
            ]

    # ── device control ───────────────────────────────────────────────

    def toggle(self, device_id: str) -> Optional[dict]:
        with self._lock:
            dev = self._devices.get(device_id)
        if not dev:
            return None

        if dev.brand == "tasmota":
            raw = self._http_get(f"http://{dev.ip}:{dev.port}/cm?cmnd=Power%20Toggle")
            if raw:
                try:
                    data = json.loads(raw)
                    power = data.get("POWER", "OFF") == "ON"
                    dev.status["power"] = power
                    return {"power": power}
                except Exception:
                    pass

        elif dev.brand == "shelly":
            raw = self._http_get(f"http://{dev.ip}:{dev.port}/relay/0?turn=toggle")
            if raw:
                try:
                    data = json.loads(raw)
                    power = data.get("ison", False)
                    dev.status["power"] = power
                    return {"power": power}
                except Exception:
                    pass

        elif dev.brand == "sonoff":
            # Sonoff eWeLink DIY mode toggle
            current = dev.status.get("power", False)
            new_state = "off" if current else "on"
            result = self._sonoff_diy_request(
                dev.ip, dev.port, "switch", {"switch": new_state}
            )
            if result and result.get("error", -1) == 0:
                dev.status["power"] = new_state == "on"
                return {"power": dev.status["power"]}
            # Fallback: try port 8081 if device is on port 80
            if dev.port != 8081:
                result = self._sonoff_diy_request(
                    dev.ip, 8081, "switch", {"switch": new_state}
                )
                if result and result.get("error", -1) == 0:
                    dev.status["power"] = new_state == "on"
                    return {"power": dev.status["power"]}

        elif dev.brand == "kasa":
            # Toggle via Kasa protocol
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(3)
                s.connect((dev.ip, 9999))
                # First get current state
                current = dev.status.get("power", False)
                new_state = 0 if current else 1
                cmd = json.dumps({"system": {"set_relay_state": {"state": new_state}}})
                key = 171
                enc = bytearray(4)
                for c in cmd:
                    b = key ^ ord(c)
                    key = b
                    enc.append(b)
                enc[:4] = struct.pack(">I", len(cmd))
                s.send(bytes(enc))
                s.recv(4096)
                s.close()
                dev.status["power"] = bool(new_state)
                return {"power": bool(new_state)}
            except Exception:
                pass

        return None

    def refresh_status(self, device_id: str) -> Optional[dict]:
        """Re-fetch status for a single device."""
        with self._lock:
            dev = self._devices.get(device_id)
        if not dev:
            return None

        if dev.brand == "tasmota":
            raw = self._http_get(f"http://{dev.ip}:{dev.port}/cm?cmnd=Status%2010")
            if raw:
                try:
                    data = json.loads(raw)
                    sns = data.get("StatusSNS", {})
                    result = {}
                    for k, v in sns.items():
                        if isinstance(v, dict):
                            if "Temperature" in v:
                                result["temperature"] = v["Temperature"]
                            if "Humidity" in v:
                                result["humidity"] = v["Humidity"]
                    # power
                    raw2 = self._http_get(f"http://{dev.ip}:{dev.port}/cm?cmnd=Power")
                    if raw2:
                        d2 = json.loads(raw2)
                        result["power"] = d2.get("POWER", "OFF") == "ON"
                    dev.status.update(result)
                    return dev.status
                except Exception:
                    pass

        elif dev.brand == "shelly":
            raw = self._http_get(f"http://{dev.ip}:{dev.port}/status")
            if raw:
                try:
                    data = json.loads(raw)
                    result = {}
                    relays = data.get("relays", [])
                    if relays:
                        result["power"] = relays[0].get("ison", False)
                    if "temperature" in data:
                        result["temperature"] = data["temperature"]
                    if "humidity" in data:
                        result["humidity"] = data["humidity"]
                    dev.status.update(result)
                    return dev.status
                except Exception:
                    pass

        elif dev.brand == "sonoff":
            port = dev.port if dev.port == 8081 else 8081
            info = self._sonoff_diy_request(dev.ip, port, "info")
            if info and "data" in info:
                d = info["data"]
                dev.status["power"] = d.get("switch", "off") == "on"
                if "signalStrength" in d:
                    dev.status["rssi"] = d["signalStrength"]
                return dev.status

        return dev.status
