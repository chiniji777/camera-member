"""Layout Manager — persistent grid layout and detection settings.

Stores layout configurations (which camera goes in which slot),
active layout, motion mode, and per-feature detection toggles.
"""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LAYOUT_FILE = Path("layout.json")

# Layout definitions: how many big/small slots each layout type has
LAYOUT_DEFS = {
    "1x1":  {"type": "grid",      "cols": 1, "total": 1,  "big": 0, "small": 0},
    "2x2":  {"type": "grid",      "cols": 2, "total": 4,  "big": 0, "small": 0},
    "3x3":  {"type": "grid",      "cols": 3, "total": 9,  "big": 0, "small": 0},
    "4x4":  {"type": "grid",      "cols": 4, "total": 16, "big": 0, "small": 0},
    "1+7":  {"type": "spotlight", "cols": 0, "total": 8,  "big": 1, "small": 7},
    "2+14": {"type": "spotlight", "cols": 0, "total": 16, "big": 2, "small": 14},
}

DEFAULT_DETECTION = {
    "motion": {"enabled": False, "sensitivity": 50},
    "human":  {"enabled": True,  "sensitivity": 50},
    "face":   {"enabled": True,  "sensitivity": 50},
    "pet":    {"enabled": False, "sensitivity": 50},
}


def _empty_slot():
    return {"camera_id": None, "name": ""}


def _build_default_layout(layout_id: str) -> dict:
    defn = LAYOUT_DEFS[layout_id]
    if defn["type"] == "grid":
        return {
            "type": "grid",
            "slots": [_empty_slot() for _ in range(defn["total"])],
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
        self.detection: dict = {}
        self._load()

    def _load(self):
        if LAYOUT_FILE.exists():
            try:
                data = json.loads(LAYOUT_FILE.read_text(encoding="utf-8"))
                self.active_layout = data.get("active_layout", "2x2")
                self.motion_mode = data.get("motion_mode", "manual")
                self.layouts = data.get("layouts", {})
                self.detection = data.get("detection", {})
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed to load layout.json: {e}")

        # Ensure all layout types exist with correct slot counts
        for layout_id in LAYOUT_DEFS:
            if layout_id not in self.layouts:
                self.layouts[layout_id] = _build_default_layout(layout_id)

        # Ensure all detection features exist
        for key, default in DEFAULT_DETECTION.items():
            if key not in self.detection:
                self.detection[key] = dict(default)

        self._save()

    def _save(self):
        data = {
            "active_layout": self.active_layout,
            "motion_mode": self.motion_mode,
            "layouts": self.layouts,
            "detection": self.detection,
        }
        LAYOUT_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def get_config(self) -> dict:
        """Return full config for the frontend."""
        return {
            "active_layout": self.active_layout,
            "motion_mode": self.motion_mode,
            "layouts": self.layouts,
            "detection": self.detection,
        }

    def set_active_layout(self, layout_id: str) -> bool:
        if layout_id not in LAYOUT_DEFS:
            return False
        self.active_layout = layout_id
        self._save()
        return True

    def update_slot(self, layout_id: str, slot_type: str, index: int,
                    camera_id: Optional[str] = None, name: Optional[str] = None) -> bool:
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
                        return False
            slot["camera_id"] = camera_id if camera_id != "" else None

        if camera_id == "":
            slot["camera_id"] = None

        if name is not None:
            slot["name"] = name

        self._save()
        return True

    def swap_slots(self, layout_id: str,
                   from_type: str, from_index: int,
                   to_type: str, to_index: int) -> bool:
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

    def update_detection(self, feature: str, enabled: Optional[bool] = None,
                         sensitivity: Optional[int] = None) -> bool:
        if feature not in self.detection:
            return False
        if enabled is not None:
            self.detection[feature]["enabled"] = enabled
        if sensitivity is not None:
            self.detection[feature]["sensitivity"] = max(0, min(100, sensitivity))
        self._save()
        return True

    def is_detection_enabled(self, feature: str) -> bool:
        return self.detection.get(feature, {}).get("enabled", False)

    def get_sensitivity(self, feature: str) -> int:
        return self.detection.get(feature, {}).get("sensitivity", 50)
