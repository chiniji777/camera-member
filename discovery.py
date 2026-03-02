"""ONVIF camera discovery via WS-Discovery multicast probe."""
import re
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
    seen_ips = set()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(
        socket.IPPROTO_IP,
        socket.IP_MULTICAST_TTL,
        struct.pack("b", 2),
    )
    sock.settimeout(timeout)

    MULTICAST_ADDR = "239.255.255.250"
    MULTICAST_PORT = 3702

    try:
        sock.sendto(ONVIF_PROBE.encode(), (MULTICAST_ADDR, MULTICAST_PORT))

        while True:
            try:
                data, addr = sock.recvfrom(65535)
                ip = addr[0]
                if ip in seen_ips:
                    continue
                seen_ips.add(ip)

                response = data.decode(errors="ignore")
                xaddr = _extract_xaddr(response)
                if xaddr:
                    cameras.append(DiscoveredCamera(
                        address=ip,
                        name=f"Camera at {ip}",
                        xaddr=xaddr,
                    ))
            except socket.timeout:
                break
    except Exception as e:
        logger.error(f"Discovery error: {e}")
    finally:
        sock.close()

    return cameras


def _extract_xaddr(xml_text: str) -> str:
    """Extract XAddrs URL from WS-Discovery response."""
    match = re.search(r"<[^>]*XAddrs[^>]*>\s*(https?://[^\s<]+)", xml_text)
    return match.group(1) if match else ""


def get_rtsp_url_from_onvif(xaddr: str) -> str:
    """Derive a best-guess RTSP URL from an ONVIF service address.

    For Hikvision cameras, the common RTSP path is:
        rtsp://<ip>:554/Streaming/Channels/101
    Channel 101 = main stream, 102 = sub stream.
    """
    match = re.search(r"https?://([\d.]+)(?::(\d+))?", xaddr)
    if match:
        ip = match.group(1)
        # Hikvision default main stream
        return f"rtsp://{ip}:554/Streaming/Channels/101"
    return ""
