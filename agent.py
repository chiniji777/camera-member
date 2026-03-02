"""
Webcam Agent — Run on any PC to share its webcam over the LAN.

Usage:
    python agent.py [--port 8554] [--camera 0]

Other devices on the LAN can view the stream at:
    http://<this-pc-ip>:<port>/video

Add that URL to the LAN Camera Viewer as an MJPEG camera.
"""
import argparse
import socket
import threading
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class MJPEGHandler(BaseHTTPRequestHandler):
    camera = None
    lock = threading.Lock()

    def do_GET(self):
        if self.path == "/video":
            self._stream_video()
        elif self.path == "/":
            self._serve_index()
        else:
            self.send_response(404)
            self.end_headers()

    def _stream_video(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            while True:
                with MJPEGHandler.lock:
                    ret, frame = MJPEGHandler.camera.read()
                if not ret:
                    continue
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                data = jpeg.tobytes()
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _serve_index(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(
            b"<html><body style='margin:0;background:#111'>"
            b"<img src='/video' style='width:100%;height:100vh;object-fit:contain'>"
            b"</body></html>"
        )

    def log_message(self, format, *args):
        pass  # Suppress per-request logs


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    parser = argparse.ArgumentParser(description="Webcam Agent — share webcam over LAN")
    parser.add_argument("--port", type=int, default=8554, help="HTTP port (default: 8554)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {args.camera}")
        return

    MJPEGHandler.camera = cap
    local_ip = get_local_ip()
    server = HTTPServer(("0.0.0.0", args.port), MJPEGHandler)

    logger.info(f"Webcam agent running")
    logger.info(f"  Local:   http://localhost:{args.port}/video")
    logger.info(f"  Network: http://{local_ip}:{args.port}/video")
    logger.info(f"Add this URL to the camera viewer as an MJPEG source.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        server.shutdown()


if __name__ == "__main__":
    main()
