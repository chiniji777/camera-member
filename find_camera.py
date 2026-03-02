"""Find Hikvision camera on LAN by scanning all IPs for common camera ports."""
import socket
import concurrent.futures

SUBNET = "192.168.1"
PORTS = [554, 8000, 80, 443, 8080, 8554]
PORT_NAMES = {554: "RTSP", 8000: "Hik-SDK", 80: "HTTP", 443: "HTTPS", 8080: "HTTP-Alt", 8554: "RTSP-Alt"}

def check_port(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    result = sock.connect_ex((ip, port))
    sock.close()
    return (ip, port, result == 0)

print(f"Scanning {SUBNET}.1-254 on ports {PORTS}...")
print("This takes about 30 seconds...\n")

found = []
with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    futures = []
    for i in range(1, 255):
        ip = f"{SUBNET}.{i}"
        for port in PORTS:
            futures.append(executor.submit(check_port, ip, port))

    for future in concurrent.futures.as_completed(futures):
        ip, port, is_open = future.result()
        if is_open:
            name = PORT_NAMES.get(port, str(port))
            print(f"  FOUND: {ip}:{port} ({name})")
            found.append((ip, port))

if found:
    # Group by IP
    ips = {}
    for ip, port in found:
        ips.setdefault(ip, []).append(port)

    print("\n--- Summary ---")
    for ip, ports in sorted(ips.items()):
        port_str = ", ".join(f"{p} ({PORT_NAMES.get(p, '')})" for p in sorted(ports))
        print(f"  {ip}: {port_str}")
        if 554 in ports or 8000 in ports:
            print(f"    ^^^ LIKELY CAMERA! Try: rtsp://admin:PASSWORD@{ip}:554/Streaming/Channels/101")
else:
    print("\nNo camera ports found. The camera might be on a different subnet or powered off.")
