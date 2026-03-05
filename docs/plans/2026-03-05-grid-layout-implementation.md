# Grid Layout Redesign — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current column-based camera grid with a slot-based layout system supporting NxN grids, spotlight layouts (1+7, 2+14), slot assignment, slot naming, drag-and-drop swap, and server-side motion detection with auto-swap.

**Architecture:** Backend gets a `LayoutManager` class for layout.json I/O, a `MotionDetector` class for frame-diff motion detection, and new API/WebSocket endpoints. Frontend replaces the current grid renderer with a layout-aware system that renders fixed slot counts, inline camera assignment dropdowns, editable slot names, and drag-and-drop between big/small slots.

**Tech Stack:** Python/FastAPI (backend), vanilla JS/CSS (frontend), WebSocket (motion events), numpy (frame diff — optional, fallback to pure Python)

---

## Task 1: LayoutManager backend class

**Files:**
- Create: `layout_manager.py`

**Step 1: Create LayoutManager class with default layouts**

```python
# layout_manager.py
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LAYOUT_FILE = Path("layout.json")

# Default layout definitions: how many big/small slots each layout has
LAYOUT_DEFS = {
    "1x1": {"type": "grid", "cols": 1, "slots": 1, "big": 0, "small": 0},
    "2x2": {"type": "grid", "cols": 2, "slots": 4, "big": 0, "small": 0},
    "3x3": {"type": "grid", "cols": 3, "slots": 9, "big": 0, "small": 0},
    "4x4": {"type": "grid", "cols": 4, "slots": 16, "big": 0, "small": 0},
    "1+7": {"type": "spotlight", "cols": 0, "slots": 8, "big": 1, "small": 7},
    "2+14": {"type": "spotlight", "cols": 0, "slots": 16, "big": 2, "small": 14},
}


def _empty_slot():
    return {"camera_id": None, "name": ""}


def _build_default_layout(layout_id: str) -> dict:
    defn = LAYOUT_DEFS[layout_id]
    if defn["type"] == "grid":
        return {
            "type": "grid",
            "slots": [_empty_slot() for _ in range(defn["slots"])],
        }
    else:
        return {
            "type": "spotlight",
            "big": [_empty_slot() for _ in range(defn["big"])],
            "small": [_empty_slot() for _ in range(defn["small"])],
        }


class LayoutManager:
    def __init__(self):
        self.active_layout: str = "2x2"
        self.motion_mode: str = "manual"  # "auto" or "manual"
        self.layouts: dict = {}
        self._load()

    def _load(self):
        if LAYOUT_FILE.exists():
            try:
                data = json.loads(LAYOUT_FILE.read_text(encoding="utf-8"))
                self.active_layout = data.get("active_layout", "2x2")
                self.motion_mode = data.get("motion_mode", "manual")
                self.layouts = data.get("layouts", {})
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed to load layout.json: {e}")

        # Ensure all layout types exist with correct slot counts
        for layout_id in LAYOUT_DEFS:
            if layout_id not in self.layouts:
                self.layouts[layout_id] = _build_default_layout(layout_id)

    def _save(self):
        data = {
            "active_layout": self.active_layout,
            "motion_mode": self.motion_mode,
            "layouts": self.layouts,
        }
        LAYOUT_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_layout(self) -> dict:
        """Return full layout config for the frontend."""
        return {
            "active_layout": self.active_layout,
            "motion_mode": self.motion_mode,
            "layouts": self.layouts,
        }

    def set_active_layout(self, layout_id: str) -> bool:
        if layout_id not in LAYOUT_DEFS:
            return False
        self.active_layout = layout_id
        self._save()
        return True

    def update_slot(self, layout_id: str, slot_type: str, index: int,
                    camera_id: Optional[str] = None, name: Optional[str] = None) -> bool:
        """Update a slot in a layout.
        slot_type: "slots" for grid layouts, "big" or "small" for spotlight layouts.
        """
        layout = self.layouts.get(layout_id)
        if not layout:
            return False

        slot_list = layout.get(slot_type)
        if not slot_list or index < 0 or index >= len(slot_list):
            return False

        slot = slot_list[index]

        # Check for duplicate camera_id across all slots in this layout
        if camera_id is not None and camera_id != "":
            for st in ["slots", "big", "small"]:
                for i, s in enumerate(layout.get(st, [])):
                    if s["camera_id"] == camera_id and not (st == slot_type and i == index):
                        return False  # duplicate
            slot["camera_id"] = camera_id if camera_id != "" else None

        if name is not None:
            slot["name"] = name

        self._save()
        return True

    def swap_slots(self, layout_id: str,
                   from_type: str, from_index: int,
                   to_type: str, to_index: int) -> bool:
        """Swap two slots (for drag-and-drop)."""
        layout = self.layouts.get(layout_id)
        if not layout:
            return False

        from_list = layout.get(from_type)
        to_list = layout.get(to_type)
        if not from_list or not to_list:
            return False
        if from_index < 0 or from_index >= len(from_list):
            return False
        if to_index < 0 or to_index >= len(to_list):
            return False

        # Swap camera_id only, keep slot names
        from_list[from_index]["camera_id"], to_list[to_index]["camera_id"] = \
            to_list[to_index]["camera_id"], from_list[from_index]["camera_id"]
        self._save()
        return True

    def set_motion_mode(self, mode: str) -> bool:
        if mode not in ("auto", "manual"):
            return False
        self.motion_mode = mode
        self._save()
        return True
```

**Step 2: Commit**

```bash
git add layout_manager.py
git commit -m "feat: add LayoutManager for persistent grid layout config"
```

---

## Task 2: Layout API endpoints in server.py

**Files:**
- Modify: `server.py:16` (add import)
- Modify: `server.py:27-32` (add layout_mgr init)
- Modify: `server.py` (add new endpoints after smart home section)

**Step 1: Add import and init**

At `server.py:16`, after the existing imports, add:
```python
from layout_manager import LayoutManager
```

At `server.py:32`, after `smart_home = SmartHomeScanner()`, add:
```python
layout_mgr = LayoutManager()
```

**Step 2: Add layout API endpoints**

Add after the smart home endpoints (after line ~389), before the static files section:

```python
# --- Layout API ---

@app.get("/api/layout")
async def get_layout():
    return layout_mgr.get_layout()


class SetActiveLayoutRequest(BaseModel):
    layout_id: str

@app.post("/api/layout/active")
async def set_active_layout(req: SetActiveLayoutRequest):
    if not layout_mgr.set_active_layout(req.layout_id):
        raise HTTPException(400, "Invalid layout ID")
    return {"ok": True, "active_layout": req.layout_id}


class UpdateSlotRequest(BaseModel):
    layout_id: str
    slot_type: str  # "slots", "big", or "small"
    index: int
    camera_id: Optional[str] = None
    name: Optional[str] = None

@app.post("/api/layout/slot")
async def update_slot(req: UpdateSlotRequest):
    if not layout_mgr.update_slot(req.layout_id, req.slot_type, req.index,
                                   camera_id=req.camera_id, name=req.name):
        raise HTTPException(400, "Invalid slot or duplicate camera")
    return {"ok": True}


class SwapSlotsRequest(BaseModel):
    layout_id: str
    from_type: str
    from_index: int
    to_type: str
    to_index: int

@app.post("/api/layout/swap")
async def swap_slots(req: SwapSlotsRequest):
    if not layout_mgr.swap_slots(req.layout_id,
                                  req.from_type, req.from_index,
                                  req.to_type, req.to_index):
        raise HTTPException(400, "Invalid swap")
    return {"ok": True}


class MotionModeRequest(BaseModel):
    mode: str  # "auto" or "manual"

@app.post("/api/motion/toggle")
async def toggle_motion_mode(req: MotionModeRequest):
    if not layout_mgr.set_motion_mode(req.mode):
        raise HTTPException(400, "Invalid mode, use 'auto' or 'manual'")
    return {"ok": True, "motion_mode": req.mode}
```

**Step 3: Commit**

```bash
git add server.py
git commit -m "feat: add layout API endpoints (get/set/slot/swap/motion-toggle)"
```

---

## Task 3: MotionDetector backend class

**Files:**
- Create: `motion_detector.py`

**Step 1: Create MotionDetector with frame-diff logic**

```python
# motion_detector.py
"""Server-side motion detection via frame pixel difference."""
import time
import struct
import logging
import threading
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Try numpy for fast frame diff; fall back to pure Python
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("numpy not installed — motion detection will use slower pure-Python fallback")


def _decode_jpeg_to_grayscale(jpeg_bytes: bytes) -> Optional[bytes]:
    """Minimal JPEG → raw grayscale pixels. Uses numpy if available."""
    if not HAS_NUMPY:
        return None
    try:
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("L")
        return img.tobytes()
    except Exception:
        return None


def _frame_diff_numpy(prev: bytes, curr: bytes, threshold: int = 30) -> float:
    """Return fraction of pixels that changed by more than threshold."""
    a = np.frombuffer(prev, dtype=np.uint8)
    b = np.frombuffer(curr, dtype=np.uint8)
    if len(a) != len(b):
        return 0.0
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    changed = np.count_nonzero(diff > threshold)
    return changed / len(a)


def _frame_diff_pure(prev: bytes, curr: bytes, threshold: int = 30) -> float:
    """Pure Python fallback — slower but works without numpy."""
    if len(prev) != len(curr):
        return 0.0
    changed = 0
    # Sample every 4th pixel for speed
    total = len(prev) // 4
    if total == 0:
        return 0.0
    for i in range(0, len(prev), 4):
        if abs(prev[i] - curr[i]) > threshold:
            changed += 1
    return changed / total


class MotionDetector:
    """Detects motion by comparing consecutive JPEG frames per camera."""

    def __init__(self, sensitivity: float = 0.05, debounce_sec: float = 2.0):
        self.sensitivity = sensitivity  # fraction of changed pixels to trigger
        self.debounce_sec = debounce_sec
        self._prev_frames: dict[str, bytes] = {}  # cam_id -> grayscale bytes
        self._last_motion: dict[str, float] = {}   # cam_id -> timestamp
        self._callbacks: list[Callable] = []
        self._enabled = False
        self._lock = threading.Lock()

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        if not enabled:
            with self._lock:
                self._prev_frames.clear()

    @property
    def enabled(self):
        return self._enabled

    def on_motion(self, callback: Callable):
        """Register callback: callback(camera_id, intensity)"""
        self._callbacks.append(callback)

    def check_frame(self, cam_id: str, jpeg_bytes: bytes):
        """Called with each new JPEG frame. Compares with previous frame."""
        if not self._enabled:
            return

        gray = _decode_jpeg_to_grayscale(jpeg_bytes)
        if gray is None:
            return

        with self._lock:
            prev = self._prev_frames.get(cam_id)
            self._prev_frames[cam_id] = gray

        if prev is None:
            return

        # Calculate difference
        if HAS_NUMPY:
            intensity = _frame_diff_numpy(prev, gray)
        else:
            intensity = _frame_diff_pure(prev, gray)

        if intensity >= self.sensitivity:
            now = time.time()
            last = self._last_motion.get(cam_id, 0)
            if now - last >= self.debounce_sec:
                self._last_motion[cam_id] = now
                logger.debug(f"Motion on {cam_id}: {intensity:.2%}")
                for cb in self._callbacks:
                    try:
                        cb(cam_id, intensity)
                    except Exception as e:
                        logger.error(f"Motion callback error: {e}")
```

**Step 2: Commit**

```bash
git add motion_detector.py
git commit -m "feat: add MotionDetector with frame-diff and numpy/PIL support"
```

---

## Task 4: Wire MotionDetector into StreamManager

**Files:**
- Modify: `camera_manager.py:8` (add import)
- Modify: `camera_manager.py:229-240` (StreamManager.__init__)
- Modify: `camera_manager.py:299-327` (_read_frames inner loop)

**Step 1: Add import at top of camera_manager.py**

After existing imports at line 8, add:
```python
from motion_detector import MotionDetector
```

**Step 2: Add motion_detector to StreamManager.__init__**

In `StreamManager.__init__()` (around line 232), add to the init body:
```python
self.motion_detector = MotionDetector()
```

**Step 3: Hook into _read_frames**

In `_read_frames`, after the line that stores the frame (line ~324):
```python
with self._frame_locks[cam_id]:
    self._frames[cam_id] = frame
```

Add immediately after:
```python
# Feed frame to motion detector
self.motion_detector.check_frame(cam_id, frame)
```

**Step 4: Commit**

```bash
git add camera_manager.py
git commit -m "feat: wire MotionDetector into StreamManager frame reader"
```

---

## Task 5: WebSocket motion endpoint

**Files:**
- Modify: `server.py` (add WebSocket import, motion wiring, WS endpoint)

**Step 1: Add WebSocket import**

At `server.py:12`, change the FastAPI import to include WebSocket:
```python
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
```

**Step 2: Wire motion detector to layout manager in lifespan**

In the `lifespan` function (around line 36), after the stream starts, add:
```python
# Wire motion detector
motion_clients = app.state.motion_clients = []
motion_lock = app.state.motion_lock = asyncio.Lock()

def on_motion(cam_id, intensity):
    """Push motion event to all connected WebSocket clients."""
    import time
    event = {"type": "motion", "camera_id": cam_id, "intensity": round(intensity, 4), "ts": time.time()}
    # Schedule async broadcast from sync context
    asyncio.get_event_loop().call_soon_threadsafe(
        asyncio.ensure_future,
        _broadcast_motion(event, motion_clients, motion_lock)
    )

stream_mgr.motion_detector.on_motion(on_motion)
# Enable motion detection based on saved mode
if layout_mgr.motion_mode == "auto":
    stream_mgr.motion_detector.set_enabled(True)
```

**Step 3: Add broadcast helper and WS endpoint**

Add near the layout API section:
```python
async def _broadcast_motion(event, clients, lock):
    import json
    msg = json.dumps(event)
    async with lock:
        dead = []
        for ws in clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.remove(ws)


@app.websocket("/ws/motion")
async def motion_ws(websocket: WebSocket):
    await websocket.accept()
    clients = app.state.motion_clients
    async with app.state.motion_lock:
        clients.append(websocket)
    try:
        while True:
            # Keep connection alive; client can send subscribe/ping
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with app.state.motion_lock:
            if websocket in clients:
                clients.remove(websocket)
```

**Step 4: Update motion toggle endpoint to enable/disable detector**

Modify the existing `toggle_motion_mode` endpoint to also enable/disable the detector:
```python
@app.post("/api/motion/toggle")
async def toggle_motion_mode(req: MotionModeRequest):
    if not layout_mgr.set_motion_mode(req.mode):
        raise HTTPException(400, "Invalid mode, use 'auto' or 'manual'")
    stream_mgr.motion_detector.set_enabled(req.mode == "auto")
    return {"ok": True, "motion_mode": req.mode}
```

**Step 5: Commit**

```bash
git add server.py
git commit -m "feat: add WebSocket /ws/motion endpoint and motion detector wiring"
```

---

## Task 6: Frontend — CSS for all layout types

**Files:**
- Modify: `static/index.html` (CSS section, lines ~56-67)

**Step 1: Replace grid CSS classes**

Replace the existing `.grid.g1` through `.grid.g4` classes (lines 64-67) with:

```css
/* --- Grid layouts (NxN) --- */
.grid.g1 { grid-template-columns: 1fr; }
.grid.g2 { grid-template-columns: 1fr 1fr; }
.grid.g3 { grid-template-columns: 1fr 1fr 1fr; }
.grid.g4 { grid-template-columns: 1fr 1fr 1fr 1fr; }

/* --- Spotlight 1+7: big left (3/5 width), 2-col small right --- */
.grid.spotlight-1-7 {
  grid-template-columns: 3fr 1fr 1fr;
  grid-template-rows: repeat(4, 1fr);
}
.grid.spotlight-1-7 .tile-big { grid-row: 1 / 4; grid-column: 1; }
.grid.spotlight-1-7 .tile-small:nth-child(2) { grid-row: 1; grid-column: 2; }
.grid.spotlight-1-7 .tile-small:nth-child(3) { grid-row: 1; grid-column: 3; }
.grid.spotlight-1-7 .tile-small:nth-child(4) { grid-row: 2; grid-column: 2; }
.grid.spotlight-1-7 .tile-small:nth-child(5) { grid-row: 2; grid-column: 3; }
.grid.spotlight-1-7 .tile-small:nth-child(6) { grid-row: 3; grid-column: 2; }
.grid.spotlight-1-7 .tile-small:nth-child(7) { grid-row: 3; grid-column: 3; }
.grid.spotlight-1-7 .tile-small:nth-child(8) { grid-row: 4; grid-column: 1 / 4; }

/* --- Spotlight 2+14: 2 bigs top-left, 7-col small grid right and below --- */
.grid.spotlight-2-14 {
  grid-template-columns: 2fr 2fr repeat(7, 1fr);
  grid-template-rows: 1fr 1fr;
}
.grid.spotlight-2-14 .tile-big:nth-child(1) { grid-row: 1 / 3; grid-column: 1; }
.grid.spotlight-2-14 .tile-big:nth-child(2) { grid-row: 1 / 3; grid-column: 2; }
/* Small tiles fill columns 3-9 across 2 rows (14 total) */

/* --- Empty slot styling --- */
.tile.empty-slot {
  border: 2px dashed #333;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  min-height: 60px;
}
.tile.empty-slot:hover { border-color: #4fc3f7; background: #1a2a3a; }
.tile.empty-slot .slot-placeholder {
  color: #555;
  font-size: 24px;
}

/* --- Slot name (editable) --- */
.tile .slot-name {
  position: absolute;
  bottom: 8px;
  left: 8px;
  background: rgba(0,0,0,0.7);
  padding: 2px 8px;
  border-radius: 3px;
  font-size: 11px;
  color: #aaa;
  cursor: text;
  border: 1px solid transparent;
  min-width: 30px;
}
.tile .slot-name:hover { border-color: #4fc3f7; }
.tile .slot-name:focus {
  outline: none;
  border-color: #4fc3f7;
  color: #fff;
}

/* --- Camera assign dropdown --- */
.slot-dropdown {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: #222;
  border: 1px solid #444;
  border-radius: 6px;
  padding: 8px 0;
  min-width: 200px;
  z-index: 100;
  max-height: 300px;
  overflow-y: auto;
}
.slot-dropdown .dd-item {
  padding: 8px 16px;
  cursor: pointer;
  font-size: 13px;
}
.slot-dropdown .dd-item:hover { background: #333; }
.slot-dropdown .dd-item.disabled { color: #555; cursor: not-allowed; }
.slot-dropdown .dd-none { padding: 12px 16px; color: #666; font-size: 13px; }

/* --- Drag and drop --- */
.tile.drag-over { outline: 2px solid #4fc3f7; outline-offset: -2px; }
.tile.dragging { opacity: 0.5; }

/* --- Motion indicator --- */
.tile .motion-badge {
  position: absolute;
  top: 8px;
  right: 40px;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #f44336;
  animation: pulse 0.6s ease-in-out infinite;
  display: none;
}
.tile .motion-badge.active { display: block; }
@keyframes pulse {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.4); opacity: 0.6; }
}

/* --- Motion mode toggle in toolbar --- */
.motion-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #888;
}
.motion-toggle .toggle-switch {
  width: 36px;
  height: 20px;
  background: #333;
  border-radius: 10px;
  position: relative;
  cursor: pointer;
  transition: background 0.2s;
}
.motion-toggle .toggle-switch.on { background: #4caf50; }
.motion-toggle .toggle-switch::after {
  content: '';
  position: absolute;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #fff;
  top: 2px;
  left: 2px;
  transition: left 0.2s;
}
.motion-toggle .toggle-switch.on::after { left: 18px; }
```

**Step 2: Commit**

```bash
git add static/index.html
git commit -m "feat: add CSS for spotlight layouts, empty slots, drag-drop, motion badge"
```

---

## Task 7: Frontend — Toolbar layout buttons update

**Files:**
- Modify: `static/index.html:850-862` (toolbar HTML)

**Step 1: Replace toolbar layout buttons**

Replace lines 853-856 with:
```html
  <button onclick="switchLayout('1x1')" id="lay-1x1">1x1</button>
  <button onclick="switchLayout('2x2')" id="lay-2x2" class="active">2x2</button>
  <button onclick="switchLayout('3x3')" id="lay-3x3">3x3</button>
  <button onclick="switchLayout('4x4')" id="lay-4x4">4x4</button>
  <button onclick="switchLayout('1+7')" id="lay-1+7">1+7</button>
  <button onclick="switchLayout('2+14')" id="lay-2+14">2+14</button>

  <div class="motion-toggle" id="motionToggle" style="display:none">
    <span>Motion</span>
    <div class="toggle-switch" id="motionSwitch" onclick="toggleMotionMode()"></div>
  </div>
```

**Step 2: Commit**

```bash
git add static/index.html
git commit -m "feat: add 1+7 and 2+14 layout buttons and motion toggle to toolbar"
```

---

## Task 8: Frontend — Layout state management JS

**Files:**
- Modify: `static/index.html` (JS section, replace grid functions starting at line ~960)

**Step 1: Replace grid JS with layout-aware system**

Replace the entire grid section (lines 960-1012, from `// --- Grid ---` through the `setInterval` camera poll) with:

```javascript
// --- Layout state ---
var layoutConfig = null;      // full layout.json from server
var activeLayout = '2x2';     // current layout ID
var motionMode = 'manual';    // 'auto' or 'manual'
var motionWs = null;          // WebSocket for motion events

async function loadLayout() {
  try {
    const res = await fetch(API + '/api/layout');
    layoutConfig = await res.json();
    activeLayout = layoutConfig.active_layout || '2x2';
    motionMode = layoutConfig.motion_mode || 'manual';
    updateToolbarButtons();
    updateMotionToggle();
    renderLayout();
  } catch (e) {
    console.error('Failed to load layout:', e);
  }
}

function switchLayout(layoutId) {
  activeLayout = layoutId;
  fetch(API + '/api/layout/active', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({layout_id: layoutId})
  });
  updateToolbarButtons();
  showMotionToggleIfSpotlight();
  renderLayout();
}

function updateToolbarButtons() {
  document.querySelectorAll('.toolbar button[id^="lay-"]').forEach(function(b) {
    b.classList.remove('active');
  });
  var btn = document.getElementById('lay-' + activeLayout);
  if (btn) btn.classList.add('active');
  showMotionToggleIfSpotlight();
}

function showMotionToggleIfSpotlight() {
  var toggle = document.getElementById('motionToggle');
  if (activeLayout === '1+7' || activeLayout === '2+14') {
    toggle.style.display = 'flex';
  } else {
    toggle.style.display = 'none';
  }
}

function updateMotionToggle() {
  var sw = document.getElementById('motionSwitch');
  if (motionMode === 'auto') {
    sw.classList.add('on');
  } else {
    sw.classList.remove('on');
  }
}

function toggleMotionMode() {
  motionMode = motionMode === 'auto' ? 'manual' : 'auto';
  updateMotionToggle();
  fetch(API + '/api/motion/toggle', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mode: motionMode})
  });
  if (motionMode === 'auto') {
    connectMotionWs();
  } else {
    disconnectMotionWs();
  }
}

// --- Camera list (for dropdown) ---
async function loadCameras() {
  try {
    const res = await fetch(API + '/api/cameras');
    cameras = await res.json();
  } catch (e) {
    console.error('Failed to load cameras:', e);
    cameras = [];
  }
}

// Poll cameras every 10s (just to keep list fresh, not to re-render grid)
setInterval(function() {
  loadCameras();
}, 10000);
```

**Step 2: Commit**

```bash
git add static/index.html
git commit -m "feat: add layout state management JS (switchLayout, motionToggle, loadLayout)"
```

---

## Task 9: Frontend — renderLayout function (grid + spotlight)

**Files:**
- Modify: `static/index.html` (replace `renderGrid` function at line ~1015)

**Step 1: Replace renderGrid with renderLayout**

Replace the `renderGrid()` function entirely with:

```javascript
// --- Render layout ---
function renderLayout() {
  var grid = document.getElementById('grid');
  while (grid.firstChild) grid.removeChild(grid.firstChild);

  if (!layoutConfig) return;

  var layout = layoutConfig.layouts[activeLayout];
  if (!layout) return;

  // Set grid CSS class
  grid.className = 'grid';
  if (layout.type === 'grid') {
    var cols = {'1x1': 1, '2x2': 2, '3x3': 3, '4x4': 4}[activeLayout] || 2;
    grid.classList.add('g' + cols);
    // Render fixed slot count
    layout.slots.forEach(function(slot, i) {
      grid.appendChild(createSlotTile(slot, 'slots', i, false));
    });
  } else if (layout.type === 'spotlight') {
    grid.classList.add(activeLayout === '1+7' ? 'spotlight-1-7' : 'spotlight-2-14');
    // Big slots first
    layout.big.forEach(function(slot, i) {
      grid.appendChild(createSlotTile(slot, 'big', i, true));
    });
    // Small slots
    layout.small.forEach(function(slot, i) {
      grid.appendChild(createSlotTile(slot, 'small', i, false));
    });
  }
}

function createSlotTile(slot, slotType, index, isBig) {
  var tile = document.createElement('div');
  tile.className = 'tile' + (isBig ? ' tile-big' : ' tile-small');
  tile.dataset.slotType = slotType;
  tile.dataset.slotIndex = index;

  // Find camera info
  var cam = slot.camera_id ? cameras.find(function(c) { return c.id === slot.camera_id; }) : null;

  if (cam) {
    // --- Occupied slot ---
    // Camera label
    var label = document.createElement('div');
    label.className = 'label';
    label.textContent = cam.name;
    tile.appendChild(label);

    // Motion badge
    var badge = document.createElement('div');
    badge.className = 'motion-badge';
    badge.id = 'motion-' + cam.id;
    tile.appendChild(badge);

    // Stream image
    var img = document.createElement('img');
    img.src = cam.stream_url;
    img.alt = cam.name;
    img.addEventListener('error', function() {
      this.style.display = 'none';
    });
    tile.appendChild(img);

    // Tracking canvas
    var canvas = document.createElement('canvas');
    canvas.className = 'tracking-canvas';
    canvas.id = 'track-' + cam.id;
    tile.appendChild(canvas);

    // Click to fullscreen
    tile.addEventListener('click', function(e) {
      if (e.target.closest('.slot-name') || e.target.closest('.slot-dropdown')) return;
      toggleFullscreen(cam.id);
    });

    // Drag support (for spotlight layouts)
    if (activeLayout === '1+7' || activeLayout === '2+14') {
      tile.draggable = true;
      tile.addEventListener('dragstart', onDragStart);
      tile.addEventListener('dragend', onDragEnd);
    }

    // Remove camera from slot (right-click)
    tile.addEventListener('contextmenu', function(e) {
      e.preventDefault();
      unassignSlot(slotType, index);
    });
  } else {
    // --- Empty slot ---
    tile.classList.add('empty-slot');
    var ph = document.createElement('div');
    ph.className = 'slot-placeholder';
    ph.textContent = '+';
    tile.appendChild(ph);

    // Click to assign camera
    tile.addEventListener('click', function(e) {
      if (e.target.closest('.slot-name')) return;
      showCameraDropdown(tile, slotType, index);
    });
  }

  // Drop target (for spotlight layouts)
  if (activeLayout === '1+7' || activeLayout === '2+14') {
    tile.addEventListener('dragover', function(e) { e.preventDefault(); tile.classList.add('drag-over'); });
    tile.addEventListener('dragleave', function() { tile.classList.remove('drag-over'); });
    tile.addEventListener('drop', function(e) { e.preventDefault(); tile.classList.remove('drag-over'); onDrop(e, slotType, index); });
  }

  // Slot name (editable)
  var nameEl = document.createElement('div');
  nameEl.className = 'slot-name';
  nameEl.contentEditable = true;
  nameEl.textContent = slot.name || '';
  nameEl.setAttribute('placeholder', 'Name...');
  nameEl.addEventListener('blur', function() {
    var newName = nameEl.textContent.trim();
    if (newName !== (slot.name || '')) {
      saveSlotName(slotType, index, newName);
    }
  });
  nameEl.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') { e.preventDefault(); nameEl.blur(); }
  });
  nameEl.addEventListener('click', function(e) { e.stopPropagation(); });
  tile.appendChild(nameEl);

  return tile;
}
```

**Step 2: Commit**

```bash
git add static/index.html
git commit -m "feat: add renderLayout with slot tiles, spotlight support, and editable names"
```

---

## Task 10: Frontend — Slot assignment dropdown + save helpers

**Files:**
- Modify: `static/index.html` (add after renderLayout)

**Step 1: Add dropdown, assign, unassign, and save functions**

```javascript
// --- Slot assignment ---
function showCameraDropdown(tile, slotType, index) {
  // Remove any existing dropdown
  var existing = document.querySelector('.slot-dropdown');
  if (existing) existing.remove();

  // Get cameras already assigned in this layout
  var layout = layoutConfig.layouts[activeLayout];
  var assignedIds = new Set();
  (layout.slots || []).forEach(function(s) { if (s.camera_id) assignedIds.add(s.camera_id); });
  (layout.big || []).forEach(function(s) { if (s.camera_id) assignedIds.add(s.camera_id); });
  (layout.small || []).forEach(function(s) { if (s.camera_id) assignedIds.add(s.camera_id); });

  var dd = document.createElement('div');
  dd.className = 'slot-dropdown';

  var available = cameras.filter(function(c) { return !assignedIds.has(c.id); });
  if (available.length === 0) {
    var none = document.createElement('div');
    none.className = 'dd-none';
    none.textContent = 'No available cameras';
    dd.appendChild(none);
  } else {
    available.forEach(function(cam) {
      var item = document.createElement('div');
      item.className = 'dd-item';
      item.textContent = cam.name;
      item.addEventListener('click', function(e) {
        e.stopPropagation();
        assignCamera(slotType, index, cam.id);
        dd.remove();
      });
      dd.appendChild(item);
    });
  }

  tile.appendChild(dd);

  // Close on outside click
  setTimeout(function() {
    document.addEventListener('click', function closeDd(e) {
      if (!dd.contains(e.target)) {
        dd.remove();
        document.removeEventListener('click', closeDd);
      }
    });
  }, 0);
}

function assignCamera(slotType, index, cameraId) {
  // Update local state
  var layout = layoutConfig.layouts[activeLayout];
  var slotList = layout[slotType];
  if (slotList && slotList[index]) {
    slotList[index].camera_id = cameraId;
  }

  // Save to server
  fetch(API + '/api/layout/slot', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      layout_id: activeLayout,
      slot_type: slotType,
      index: index,
      camera_id: cameraId
    })
  });

  renderLayout();
}

function unassignSlot(slotType, index) {
  var layout = layoutConfig.layouts[activeLayout];
  var slotList = layout[slotType];
  if (slotList && slotList[index]) {
    slotList[index].camera_id = null;
  }

  fetch(API + '/api/layout/slot', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      layout_id: activeLayout,
      slot_type: slotType,
      index: index,
      camera_id: ''
    })
  });

  renderLayout();
}

function saveSlotName(slotType, index, name) {
  var layout = layoutConfig.layouts[activeLayout];
  var slotList = layout[slotType];
  if (slotList && slotList[index]) {
    slotList[index].name = name;
  }

  fetch(API + '/api/layout/slot', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      layout_id: activeLayout,
      slot_type: slotType,
      index: index,
      name: name
    })
  });
}
```

**Step 2: Commit**

```bash
git add static/index.html
git commit -m "feat: add camera assignment dropdown, unassign, and slot name save"
```

---

## Task 11: Frontend — Drag and drop swap

**Files:**
- Modify: `static/index.html` (add after slot assignment section)

**Step 1: Add drag-and-drop handlers**

```javascript
// --- Drag and drop (spotlight layouts) ---
var dragSource = null;

function onDragStart(e) {
  dragSource = {
    type: e.currentTarget.dataset.slotType,
    index: parseInt(e.currentTarget.dataset.slotIndex)
  };
  e.currentTarget.classList.add('dragging');
  e.dataTransfer.effectAllowed = 'move';
}

function onDragEnd(e) {
  e.currentTarget.classList.remove('dragging');
  dragSource = null;
}

function onDrop(e, targetType, targetIndex) {
  if (!dragSource) return;
  if (dragSource.type === targetType && dragSource.index === targetIndex) return;

  // Swap locally
  var layout = layoutConfig.layouts[activeLayout];
  var fromList = layout[dragSource.type];
  var toList = layout[targetType];
  if (!fromList || !toList) return;

  var tmpCamId = fromList[dragSource.index].camera_id;
  fromList[dragSource.index].camera_id = toList[targetIndex].camera_id;
  toList[targetIndex].camera_id = tmpCamId;

  // Save to server
  fetch(API + '/api/layout/swap', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      layout_id: activeLayout,
      from_type: dragSource.type,
      from_index: dragSource.index,
      to_type: targetType,
      to_index: targetIndex
    })
  });

  dragSource = null;
  renderLayout();
}
```

**Step 2: Commit**

```bash
git add static/index.html
git commit -m "feat: add drag-and-drop swap between big/small slots"
```

---

## Task 12: Frontend — WebSocket motion events + auto-swap

**Files:**
- Modify: `static/index.html` (add after drag-and-drop section)

**Step 1: Add WebSocket motion client and auto-swap logic**

```javascript
// --- Motion detection WebSocket ---
function connectMotionWs() {
  if (motionWs) return;
  var wsUrl = 'ws://' + location.host + '/ws/motion';
  motionWs = new WebSocket(wsUrl);
  motionWs.onmessage = function(e) {
    try {
      var event = JSON.parse(e.data);
      if (event.type === 'motion') {
        handleMotionEvent(event.camera_id, event.intensity);
      }
    } catch (err) {}
  };
  motionWs.onclose = function() {
    motionWs = null;
    // Reconnect if still in auto mode
    if (motionMode === 'auto') {
      setTimeout(connectMotionWs, 3000);
    }
  };
  motionWs.onopen = function() {
    motionWs.send(JSON.stringify({type: 'subscribe'}));
  };
}

function disconnectMotionWs() {
  if (motionWs) {
    motionWs.close();
    motionWs = null;
  }
}

var lastAutoSwap = {};  // camera_id -> timestamp (cooldown)

function handleMotionEvent(cameraId, intensity) {
  // Flash motion badge
  var badge = document.getElementById('motion-' + cameraId);
  if (badge) {
    badge.classList.add('active');
    setTimeout(function() { badge.classList.remove('active'); }, 2000);
  }

  // Auto-swap only in spotlight layouts with auto mode
  if (motionMode !== 'auto') return;
  if (activeLayout !== '1+7' && activeLayout !== '2+14') return;

  var layout = layoutConfig.layouts[activeLayout];
  if (!layout || layout.type !== 'spotlight') return;

  // Check if this camera is already in a big slot
  var alreadyBig = layout.big.some(function(s) { return s.camera_id === cameraId; });
  if (alreadyBig) return;

  // Find which small slot has this camera
  var smallIdx = -1;
  layout.small.forEach(function(s, i) {
    if (s.camera_id === cameraId) smallIdx = i;
  });
  if (smallIdx === -1) return;

  // Cooldown check (10 seconds)
  var now = Date.now();
  if (lastAutoSwap[cameraId] && now - lastAutoSwap[cameraId] < 10000) return;
  lastAutoSwap[cameraId] = now;

  // Find the big slot to swap into (oldest last-motion or first big slot)
  var targetBigIdx = 0;
  if (layout.big.length > 1) {
    // Pick the big slot whose camera had the oldest motion
    var oldestTime = Infinity;
    layout.big.forEach(function(s, i) {
      var t = lastAutoSwap[s.camera_id] || 0;
      if (t < oldestTime) { oldestTime = t; targetBigIdx = i; }
    });
  }

  // Swap
  var tmpCamId = layout.big[targetBigIdx].camera_id;
  layout.big[targetBigIdx].camera_id = cameraId;
  layout.small[smallIdx].camera_id = tmpCamId;

  fetch(API + '/api/layout/swap', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      layout_id: activeLayout,
      from_type: 'small',
      from_index: smallIdx,
      to_type: 'big',
      to_index: targetBigIdx
    })
  });

  renderLayout();
}
```

**Step 2: Commit**

```bash
git add static/index.html
git commit -m "feat: add WebSocket motion client with auto-swap for spotlight layouts"
```

---

## Task 13: Frontend — Wire up initialization

**Files:**
- Modify: `static/index.html` (replace the init call at the bottom of the script)

**Step 1: Update initialization to load layout first**

Find the existing `loadCameras()` call at the bottom of the script and replace with:

```javascript
// --- Init ---
async function init() {
  await loadCameras();
  await loadLayout();
  if (motionMode === 'auto') {
    connectMotionWs();
  }
}
init();
```

Also remove the old `autoGrid` and `setGrid` functions entirely since they're replaced by `switchLayout`.

Remove the old `setInterval` camera poll that calls `autoGrid(cameras.length); renderGrid();` — the new poll in Task 8 just refreshes the `cameras` array without re-rendering (the layout drives rendering now).

**Step 2: Commit**

```bash
git add static/index.html
git commit -m "feat: wire up init to load layout then cameras, connect motion WS"
```

---

## Task 14: Install dependencies

**Files:**
- Modify: `requirements.txt` (if exists) or note in README

**Step 1: Install Pillow and numpy for motion detection**

```bash
pip install Pillow numpy
```

These are needed by `motion_detector.py` for frame-to-grayscale conversion and fast pixel diff.

**Step 2: Commit (if requirements.txt exists)**

```bash
pip freeze | grep -iE "pillow|numpy" >> requirements.txt
git add requirements.txt
git commit -m "chore: add Pillow and numpy for motion detection"
```

---

## Task 15: Integration test — manual walkthrough

**Step 1: Start the server**

```bash
python server.py
```

**Step 2: Verify layout API**

```bash
curl http://localhost:8080/api/layout
```
Expected: JSON with `active_layout`, `motion_mode`, and `layouts` object containing all 6 layout types.

**Step 3: Test in browser**

Open `http://localhost:8080` and verify:
1. Toolbar shows 6 layout buttons: 1x1, 2x2, 3x3, 4x4, 1+7, 2+14
2. Click 2x2 → shows 4 slots (some empty with `+` icon)
3. Click empty slot → dropdown shows available cameras
4. Assign camera → stream appears in slot
5. Click slot name area → edit name, press Enter to save
6. Refresh page → layout and names persist
7. Right-click occupied slot → camera removed from slot
8. Click 1+7 → shows 1 big + 7 small layout
9. Motion toggle appears for spotlight layouts
10. Drag camera from small to big slot → swap works

**Step 4: Commit all remaining changes**

```bash
git add -A
git commit -m "feat: grid layout redesign — slot assignment, spotlight layouts, motion detect"
```
