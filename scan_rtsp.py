"""Quick scan: find devices with RTSP port 554 open on the LAN."""
import socket

ips = [
    "192.168.1.110", "192.168.1.112", "192.168.1.115",
    "192.168.1.116", "192.168.1.117", "192.168.1.118",
    "192.168.1.120", "192.168.1.122", "192.168.1.124",
    "192.168.1.126",
]

for ip in ips:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex((ip, 554))
    if result == 0:
        print(f"RTSP OPEN: {ip}:554")
    # Also check port 80 (web interface)
    sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock2.settimeout(1)
    result2 = sock2.connect_ex((ip, 80))
    if result2 == 0:
        print(f"HTTP OPEN: {ip}:80")
    sock.close()
    sock2.close()

print("Scan done.")
