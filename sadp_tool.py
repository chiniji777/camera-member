"""
SADP Tool — Discover and activate Hikvision cameras on the LAN.

Hikvision cameras use SADP (Search Active Devices Protocol) over
UDP port 37020 for device discovery and initial activation.

Usage:
    python sadp_tool.py discover         # Find cameras
    python sadp_tool.py activate <ip> <password>  # Activate a camera
"""
import socket
import struct
import sys
import uuid
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime

SADP_PORT = 37020
SADP_MULTICAST = "239.255.255.250"


def build_probe_packet():
    """Build SADP inquiry/probe packet.

    SADP packet structure:
    - 8 bytes: header (magic + version + type)
    - 24 bytes: padding/counters
    - Followed by XML probe body
    """
    probe_uuid = str(uuid.uuid4())
    xml_body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Probe>'
        f'<Uuid>{probe_uuid}</Uuid>'
        '<Types>inquiry</Types>'
        '</Probe>'
    ).encode('utf-8')

    # SADP header: 8 bytes
    # Byte 0: 0x21 (packet start marker)
    # Byte 1: 0x02 (version)
    # Byte 2-3: 0x01 0x00 (type = inquiry)
    # Byte 4-7: length of xml body (little-endian)
    header = struct.pack('<BBBB I', 0x21, 0x02, 0x01, 0x00, len(xml_body))

    # Counter/sequence padding (24 bytes)
    padding = b'\x00' * 24

    return header + padding + xml_body


def build_probe_packet_v2():
    """Alternative simpler probe - just the raw SADP inquiry bytes.

    Some cameras respond to a minimal 32-byte inquiry packet.
    """
    # Minimal SADP inquiry packet (known working format)
    packet = bytearray(32)
    packet[0] = 0x21   # Magic
    packet[1] = 0x02   # Version
    packet[2] = 0x01   # Type: inquiry
    packet[3] = 0x00
    return bytes(packet)


def discover(timeout=5.0):
    """Send SADP probes and collect responses."""
    cameras = []

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(timeout)

    # Try to join multicast group
    try:
        mreq = struct.pack("4sl", socket.inet_aton(SADP_MULTICAST), socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    except Exception as e:
        print(f"  Warning: Could not join multicast group: {e}")

    # Send both probe formats
    probe1 = build_probe_packet()
    probe2 = build_probe_packet_v2()

    targets = [
        (SADP_MULTICAST, SADP_PORT),
        ("255.255.255.255", SADP_PORT),
        ("192.168.1.255", SADP_PORT),
    ]

    for target in targets:
        try:
            sock.sendto(probe1, target)
            sock.sendto(probe2, target)
        except Exception as e:
            print(f"  Warning: Failed to send to {target}: {e}")

    # Also send directly to known camera IP if we found one
    try:
        sock.sendto(probe1, ("192.168.1.123", SADP_PORT))
        sock.sendto(probe2, ("192.168.1.123", SADP_PORT))
    except Exception:
        pass

    print(f"  Listening for responses ({timeout}s timeout)...")
    seen = set()

    while True:
        try:
            data, addr = sock.recvfrom(65535)
            ip = addr[0]
            if ip in seen:
                continue
            seen.add(ip)

            info = parse_response(data, ip)
            if info:
                cameras.append(info)
                print(f"  Found: {info}")
        except socket.timeout:
            break
        except Exception as e:
            print(f"  Error receiving: {e}")

    sock.close()
    return cameras


def parse_response(data, sender_ip):
    """Parse SADP response packet."""
    info = {"ip": sender_ip, "raw_len": len(data)}

    # Try to find XML in the response
    try:
        # Look for XML start
        xml_start = data.find(b'<?xml')
        if xml_start == -1:
            xml_start = data.find(b'<')

        if xml_start >= 0:
            xml_data = data[xml_start:].decode('utf-8', errors='ignore')
            root = ET.fromstring(xml_data)

            # Extract common fields
            ns = {'hik': 'http://www.hikvision.com/ver20/XMLSchema'}
            for child in root.iter():
                tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if child.text and child.text.strip():
                    info[tag] = child.text.strip()

            return info
        else:
            # Binary response - try to extract info from fixed positions
            if len(data) >= 32:
                info["type"] = "binary_response"
                # Try to extract IP from bytes (common in older SADP)
                if len(data) >= 48:
                    try:
                        ip_bytes = data[16:20]
                        extracted_ip = ".".join(str(b) for b in ip_bytes)
                        if extracted_ip.startswith("192.168"):
                            info["device_ip"] = extracted_ip
                    except Exception:
                        pass
                return info
    except ET.ParseError:
        info["parse_error"] = True
        # Store raw text portion for debugging
        text_part = data[data.find(b'<'):].decode('utf-8', errors='ignore')[:200] if b'<' in data else ""
        if text_part:
            info["raw_xml"] = text_part
        return info
    except Exception as e:
        info["error"] = str(e)
        return info

    return None


def activate_via_sadp(camera_ip, password):
    """Attempt to activate camera via SADP activation packet."""
    print(f"\nActivating camera at {camera_ip}...")

    # Method 1: SADP activation packet
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(5)

    activate_uuid = str(uuid.uuid4())
    xml_body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Activate>'
        f'<Uuid>{activate_uuid}</Uuid>'
        '<Types>activate</Types>'
        f'<Password>{password}</Password>'
        '</Activate>'
    ).encode('utf-8')

    header = struct.pack('<BBBB I', 0x21, 0x02, 0x03, 0x00, len(xml_body))
    padding = b'\x00' * 24
    packet = header + padding + xml_body

    try:
        sock.sendto(packet, (camera_ip, SADP_PORT))
        print("  Sent SADP activation packet...")

        try:
            data, addr = sock.recvfrom(65535)
            print(f"  Response from {addr}: {len(data)} bytes")
            info = parse_response(data, addr[0])
            if info:
                print(f"  Response: {info}")
        except socket.timeout:
            print("  No SADP response (may still have worked)")
    except Exception as e:
        print(f"  SADP error: {e}")
    finally:
        sock.close()

    # Method 2: Try ISAPI activation with digest auth
    print("\n  Trying ISAPI activation...")
    try:
        import urllib.request
        import urllib.error

        activate_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f'<ActivateInfo><password>{password}</password></ActivateInfo>'
        )

        # Try PUT
        req = urllib.request.Request(
            f"http://{camera_ip}/ISAPI/Security/activate",
            data=activate_xml.encode('utf-8'),
            method='PUT',
            headers={'Content-Type': 'application/xml'}
        )
        try:
            resp = urllib.request.urlopen(req, timeout=5)
            result = resp.read().decode()
            print(f"  ISAPI PUT response: {result[:300]}")
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', errors='ignore')
            print(f"  ISAPI PUT HTTP {e.code}: {body[:300]}")

        # Try POST as fallback
        req2 = urllib.request.Request(
            f"http://{camera_ip}/ISAPI/Security/activate",
            data=activate_xml.encode('utf-8'),
            method='POST',
            headers={'Content-Type': 'application/xml'}
        )
        try:
            resp2 = urllib.request.urlopen(req2, timeout=5)
            result2 = resp2.read().decode()
            print(f"  ISAPI POST response: {result2[:300]}")
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', errors='ignore')
            print(f"  ISAPI POST HTTP {e.code}: {body[:300]}")

    except Exception as e:
        print(f"  ISAPI error: {e}")

    # Verify activation
    print("\n  Verifying activation status...")
    try:
        req = urllib.request.Request(f"http://{camera_ip}/ISAPI/System/deviceInfo")
        resp = urllib.request.urlopen(req, timeout=5)
        result = resp.read().decode()
        if "notActivated" in result:
            print("  Camera is still NOT activated.")
        else:
            print("  Camera appears to be activated!")
            print(f"  Response: {result[:300]}")
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("  Camera is ACTIVATED! (returns 401 = needs authentication now)")
            return True
        body = e.read().decode('utf-8', errors='ignore')
        if "notActivated" in body:
            print("  Camera is still NOT activated.")
        else:
            print(f"  HTTP {e.code}: {body[:300]}")
    except Exception as e:
        print(f"  Verify error: {e}")

    return False


def main():
    if len(sys.argv) < 2:
        print("SADP Tool for Hikvision Cameras")
        print("================================")
        print("Usage:")
        print("  python sadp_tool.py discover")
        print("  python sadp_tool.py activate <ip> <password>")
        print("  python sadp_tool.py status <ip>")
        return

    cmd = sys.argv[1]

    if cmd == "discover":
        print("Discovering Hikvision cameras via SADP...\n")
        cameras = discover(timeout=5)
        if cameras:
            print(f"\nFound {len(cameras)} device(s)")
        else:
            print("\nNo SADP responses received.")
            print("Tip: Windows Firewall may block UDP 37020. Try running as administrator.")

    elif cmd == "activate":
        if len(sys.argv) < 4:
            print("Usage: python sadp_tool.py activate <ip> <password>")
            return
        ip = sys.argv[2]
        password = sys.argv[3]
        success = activate_via_sadp(ip, password)
        if success:
            print(f"\nCamera activated! RTSP URL:")
            print(f"  rtsp://admin:{password}@{ip}:554/Streaming/Channels/101")

    elif cmd == "status":
        if len(sys.argv) < 3:
            print("Usage: python sadp_tool.py status <ip>")
            return
        ip = sys.argv[2]
        print(f"Checking activation status of {ip}...")
        try:
            import urllib.request
            import urllib.error
            req = urllib.request.Request(f"http://{ip}/ISAPI/System/deviceInfo")
            resp = urllib.request.urlopen(req, timeout=5)
            print("Camera is activated (returned device info)")
        except urllib.error.HTTPError as e:
            if e.code == 401:
                print("Camera is ACTIVATED (requires auth)")
            else:
                body = e.read().decode('utf-8', errors='ignore')
                if "notActivated" in body:
                    print("Camera is NOT ACTIVATED")
                else:
                    print(f"HTTP {e.code}: {body[:200]}")
        except Exception as e:
            print(f"Error: {e}")

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
