"""
Gesture classifier.

Phase 1 (current): Rule-based geometry on the 21 MediaPipe hand landmarks.
                   Works out of the box — no training data needed.
Phase 2 (upgrade): Drop in a trained TensorFlow model by un-commenting
                   the CNN section and calling model.predict().

Landmark index reference (MediaPipe Hands):
  0  = WRIST
  4  = THUMB_TIP
  8  = INDEX_FINGER_TIP
  12 = MIDDLE_FINGER_TIP
  16 = RING_FINGER_TIP
  20 = PINKY_TIP
  MCP joints: 5, 9, 13, 17
  PIP joints: 6, 10, 14, 18
"""

import math
import os
import numpy as np

# ── optional CNN model ────────────────────────────────────────────────────────
_MODEL = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), "gesture_model.h5")

def _load_model():
    global _MODEL
    if os.path.exists(MODEL_PATH):
        try:
            from tensorflow import keras          # noqa: PLC0415
            _MODEL = keras.models.load_model(MODEL_PATH)
            print("[ML] Loaded gesture_model.h5")
        except Exception as exc:
            print(f"[ML] Could not load model: {exc}")

_load_model()

LABELS = [
    "hello", "help", "water", "food", "pain", "bathroom",
    "yes", "no", "thanks", "stop", "doctor", "tired",
    "love", "good", "bad", "call",
]

# ── geometry helpers ──────────────────────────────────────────────────────────

def _pt(lm, idx):
    """Return (x, y, z) for landmark index."""
    p = lm[idx]
    if isinstance(p, dict):
        return p["x"], p["y"], p.get("z", 0)
    return float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0


def _dist(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _finger_extended(lm, tip, pip, mcp, wrist):
    """True if finger tip is farther from wrist than MCP (finger is open)."""
    tip_d = _dist(_pt(lm, tip),   _pt(lm, wrist))
    mcp_d = _dist(_pt(lm, mcp),   _pt(lm, wrist))
    return tip_d > mcp_d * 1.1


def _fingers_open(lm):
    wrist = 0
    return {
        "thumb":  _finger_extended(lm, 4,  3,  2,  wrist),
        "index":  _finger_extended(lm, 8,  6,  5,  wrist),
        "middle": _finger_extended(lm, 12, 10, 9,  wrist),
        "ring":   _finger_extended(lm, 16, 14, 13, wrist),
        "pinky":  _finger_extended(lm, 20, 18, 17, wrist),
    }

# ── rule-based classifier ─────────────────────────────────────────────────────

def _rule_classify(lm):
    fo = _fingers_open(lm)
    n_open = sum(fo.values())

    # YES  — fist (all closed)
    if n_open == 0:
        return "yes", 0.92

    # ALL OPEN → hello or stop
    if n_open == 5:
        # thumb tip y < wrist y  (hand upright) → hello wave
        if _pt(lm, 4)[1] < _pt(lm, 0)[1]:
            return "hello", 0.88
        return "stop", 0.85

    # ONLY INDEX → pointing → help
    if fo["index"] and not fo["middle"] and not fo["ring"] and not fo["pinky"]:
        return "help", 0.87

    # INDEX + MIDDLE (peace/V) → no
    if fo["index"] and fo["middle"] and not fo["ring"] and not fo["pinky"]:
        return "no", 0.89

    # THUMB + PINKY (shaka) → call
    if fo["thumb"] and fo["pinky"] and not fo["index"] and not fo["middle"] and not fo["ring"]:
        return "call", 0.86

    # THUMB ONLY → love (heart)
    if fo["thumb"] and not fo["index"] and not fo["middle"] and not fo["ring"] and not fo["pinky"]:
        return "love", 0.82

    # PINKY ONLY → water (ASL W approximation)
    if fo["pinky"] and not fo["index"] and not fo["middle"] and not fo["ring"] and not fo["thumb"]:
        return "water", 0.80

    # 3 fingers (index+middle+ring) → doctor
    if fo["index"] and fo["middle"] and fo["ring"] and not fo["pinky"]:
        return "doctor", 0.81

    # 4 fingers (no thumb) → bathroom
    if fo["index"] and fo["middle"] and fo["ring"] and fo["pinky"] and not fo["thumb"]:
        return "bathroom", 0.83

    # Thumb + index (gun shape) → pain
    if fo["thumb"] and fo["index"] and not fo["middle"] and not fo["ring"] and not fo["pinky"]:
        return "pain", 0.84

    # Thumb + index + middle → thanks (ASL flat hand near chin)
    if fo["thumb"] and fo["index"] and fo["middle"] and not fo["ring"] and not fo["pinky"]:
        return "thanks", 0.79

    # Default — tired (closed, relaxed)
    return "tired", 0.60


# ── public API ────────────────────────────────────────────────────────────────

def classify_landmarks(landmarks):
    """
    landmarks: list of 21 dicts  {"x": float, "y": float, "z": float}
               or list of 3-element lists [[x,y,z], ...]

    Returns (gesture_label: str, confidence: float 0-1)
    """
    if _MODEL is not None:
        # CNN path
        flat = []
        for p in landmarks:
            if isinstance(p, dict):
                flat.extend([p["x"], p["y"], p.get("z", 0)])
            else:
                flat.extend(p[:3])
        x = np.array(flat, dtype=np.float32).reshape(1, -1)
        probs = _MODEL.predict(x, verbose=0)[0]
        idx   = int(np.argmax(probs))
        return LABELS[idx], float(probs[idx])

    # Rule-based path
    return _rule_classify(landmarks)
