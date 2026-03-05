# Grid Layout Redesign — Design Doc

**Date**: 2026-03-05
**Status**: Approved

## Problem

Current grid system treats NxN as column count (2x2 = 2 columns, shows only existing cameras). User expects 2x2 = 4 slots. No way to assign cameras to specific slots or name them.

## Requirements

1. NxN grids always show N*N slots (2x2=4, 3x3=9, 4x4=16)
2. New layouts: 1+7 (1 big + 7 small) and 2+14 (2 big + 14 small)
3. Click empty slot to assign a camera from dropdown
4. Name each slot (editable inline)
5. No duplicate cameras across slots
6. Layout saved permanently to server-side JSON
7. Motion detection: auto-swap camera with motion to big slot
8. Motion detect toggleable: auto mode vs manual drag-swap mode

## Layout Types

| Mode | Big | Small | Total |
|------|-----|-------|-------|
| 1x1  | 1   | 0     | 1     |
| 2x2  | 0   | 4     | 4     |
| 3x3  | 0   | 9     | 9     |
| 4x4  | 0   | 16    | 16    |
| 1+7  | 1   | 7     | 8     |
| 2+14 | 2   | 14    | 16    |

## Layout Mockups

### 1+7
```
+----------------+------+------+
|                |  S1  |  S2  |
|     BIG 1      +------+------+
|                |  S3  |  S4  |
|                +------+------+
|                |  S5  |  S6  |
+----------------+------+------+
                    S7
```

### 2+14
```
+----------+----------+---+---+---+---+---+---+---+
|          |          | S1| S2| S3| S4| S5| S6| S7|
|  BIG 1   |  BIG 2   +---+---+---+---+---+---+---+
|          |          | S8| S9|S10|S11|S12|S13|S14|
+----------+----------+---+---+---+---+---+---+---+
```

## Data Model — `layout.json`

```json
{
  "active_layout": "1+7",
  "motion_mode": "auto",
  "layouts": {
    "1x1": {
      "type": "grid",
      "slots": [
        {"camera_id": "d7b267c4", "name": "Main"}
      ]
    },
    "2x2": {
      "type": "grid",
      "slots": [
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""}
      ]
    },
    "1+7": {
      "type": "spotlight",
      "big": [{"camera_id": null, "name": ""}],
      "small": [
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""}
      ]
    },
    "2+14": {
      "type": "spotlight",
      "big": [
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""}
      ],
      "small": [
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""},
        {"camera_id": null, "name": ""}
      ]
    }
  }
}
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET    | `/api/layout` | Load layout config |
| POST   | `/api/layout` | Save layout config |
| WS     | `/ws/motion` | Motion detection event stream |
| POST   | `/api/motion/toggle` | Switch auto/manual mode |

## Motion Detection

### Server-side algorithm
1. In `StreamManager._read_frames()`, keep previous frame per camera
2. Convert current + previous frame to grayscale (simple average)
3. Calculate pixel-wise absolute difference
4. Count pixels exceeding threshold (default: 30/255)
5. If changed pixel % > sensitivity threshold (default: 5%) → motion event
6. Debounce: min 2 seconds between events per camera

### WebSocket protocol
```json
// Server → Client
{"type": "motion", "camera_id": "d7b267c4", "intensity": 0.12, "ts": 1709654400}

// Client → Server
{"type": "subscribe"}
```

### Auto mode behavior
- Motion detected on small-slot camera → swap it to big slot
- The camera previously in the big slot moves to the small slot
- If multiple big slots (2+14): fill the one with oldest last-motion time
- Cooldown: don't auto-swap the same camera back within 10 seconds

### Manual mode behavior
- Motion detection disabled (no frame comparison)
- User drags camera tiles between big and small slots
- HTML5 drag/drop with visual highlight on drop targets

## Slot Assignment UI

- Click empty slot → inline dropdown of available (unassigned) cameras
- Click slot name → inline text edit (press Enter to save)
- Right-click slot → context menu: "Remove camera", "Rename"

## Architecture

### Frontend changes (index.html)
- Replace current grid CSS/JS with layout-aware renderer
- Add layout selector buttons: 1x1, 2x2, 3x3, 4x4, 1+7, 2+14
- Add motion mode toggle button (auto/manual)
- Add drag-and-drop handlers for spotlight layouts
- Add WebSocket client for motion events
- Add slot assignment dropdown and name editing

### Backend changes (server.py)
- Add `/api/layout` GET/POST endpoints
- Add `/ws/motion` WebSocket endpoint
- Add `/api/motion/toggle` endpoint

### Backend changes (camera_manager.py)
- Add `MotionDetector` class with frame comparison logic
- Add motion event callback system
- Add `LayoutManager` class for layout.json I/O

### New file: `layout.json`
- Stores all layout configurations and active layout
