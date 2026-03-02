# LAN Camera Viewer — Design Document

**Date:** 2026-03-02
**Status:** Approved

## Overview

A web-based application for viewing up to 16 camera feeds from IP cameras (RTSP/ONVIF) and shared webcams on a WiFi LAN. Live viewing only, no recording.

## Architecture

Three components:

1. **Main Server** — Python FastAPI backend that discovers cameras, manages FFmpeg processes to convert RTSP to HLS, and serves the web UI.
2. **Webcam Agent** — Standalone Python script (~50 lines) that captures a local webcam via OpenCV and serves MJPEG over HTTP. Runs on any PC with a webcam.
3. **Web UI** — Single-page HTML/JS app with configurable grid (1x1 to 4x4), dark theme. Uses hls.js for HLS streams and native `<img>` for MJPEG.

## Tech Stack

- Python 3.10+, FastAPI, uvicorn
- FFmpeg (subprocess, no re-encoding — video passthrough)
- onvif-zeep for ONVIF camera discovery
- OpenCV for webcam agent
- hls.js (CDN) for browser HLS playback
- Vanilla HTML/CSS/JS — no frontend framework

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve web UI |
| `/api/cameras` | GET | List all cameras |
| `/api/cameras` | POST | Add a camera (RTSP or agent URL) |
| `/api/cameras/{id}` | DELETE | Remove a camera |
| `/api/discover` | POST | Scan LAN for ONVIF cameras |
| `/streams/{id}/` | GET | HLS playlist (.m3u8) |
| `/streams/{id}/{segment}` | GET | HLS segments (.ts) |

## Stream Pipeline

```
RTSP source → FFmpeg subprocess → HLS segments (3s, 3-segment buffer)
                                  written to temp dir → served by FastAPI
```

- One FFmpeg process per camera
- FFmpeg args: `-rtsp_transport tcp -i {url} -c:v copy -f hls -hls_time 3 -hls_list_size 3 -hls_flags delete_segments`
- Auto-restart with exponential backoff on crash (max 30s)

## Webcam Agent

- Single file: `agent.py`
- Usage: `python agent.py --port 8554 --camera 0`
- Captures via OpenCV, serves MJPEG at `http://{ip}:{port}/video`
- Main server treats agent URLs like any other camera

## Web UI

- Grid layouts: 1x1, 2x2, 3x3, 4x4 selectable via toolbar
- Camera tiles with name overlay
- Add camera panel (manual RTSP/agent URL)
- Discover button (ONVIF scan)
- Click-to-fullscreen on any tile
- Auto-reconnect with "Offline" overlay on stream drop
- Dark theme, CSS Grid layout

## Error Handling

- Camera offline: tile shows "Offline" + red dot, retries every 10s
- FFmpeg crash: auto-restart with exponential backoff
- Agent unreachable: same offline treatment
- Port conflict: log error, suggest alternative

## Persistence

Simple `cameras.json` file — no database.

## Project Structure

```
Camera member/
├── server.py          # Main FastAPI server
├── camera_manager.py  # FFmpeg process management, camera CRUD
├── discovery.py       # ONVIF LAN scanner
├── cameras.json       # Persisted camera config (auto-created)
├── agent.py           # Webcam agent (run on other PCs)
├── requirements.txt   # Python dependencies
├── static/
│   └── index.html     # Web UI (single file with CSS + JS)
└── streams/           # Temp dir for HLS segments (auto-created)
```

## Dependencies

```
fastapi
uvicorn
onvif-zeep
opencv-python
```

hls.js loaded via CDN in the HTML.
