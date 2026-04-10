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
    "hello", "yes", "no", "help", "stop", "water",
    "pain", "call", "doctor", "bathroom", "thanks",
    "did_you_eat",
]

# Tuned thresholds for webcam-scale normalized landmarks.
MOTION_HELLO_MIN = 0.014
PINCH_PAIN_MIN = 0.34
PEACE_GAP_MIN = 0.18
DID_YOU_EAT_THUMB_GAP_MIN = 0.22

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
    """True if finger is open based on distance + joint angle hints."""
    tip_d = _dist(_pt(lm, tip), _pt(lm, wrist))
    mcp_d = _dist(_pt(lm, mcp), _pt(lm, wrist))
    pip_d = _dist(_pt(lm, pip), _pt(lm, wrist))
    return tip_d > max(mcp_d * 1.12, pip_d * 1.06)


def _fingers_open(lm):
    wrist = 0
    return {
        "thumb":  _finger_extended(lm, 4,  3,  2,  wrist),
        "index":  _finger_extended(lm, 8,  6,  5,  wrist),
        "middle": _finger_extended(lm, 12, 10, 9,  wrist),
        "ring":   _finger_extended(lm, 16, 14, 13, wrist),
        "pinky":  _finger_extended(lm, 20, 18, 17, wrist),
    }


def _all_closed_except(fo, keep_open):
    return all((name in keep_open) == is_open for name, is_open in fo.items())

# ── rule-based classifier ─────────────────────────────────────────────────────

def _rule_classify(lm, motion=0.0):
    fo = _fingers_open(lm)
    n_open = sum(fo.values())
    thumb_tip = _pt(lm, 4)
    index_tip = _pt(lm, 8)
    index_mcp = _pt(lm, 5)
    middle_tip = _pt(lm, 12)
    wrist = _pt(lm, 0)

    pinch_dist = _dist(thumb_tip, index_tip)
    palm_scale = max(_dist(wrist, _pt(lm, 9)), 1e-6)
    pinch_ratio = pinch_dist / palm_scale

    thumb_index_gap = abs(thumb_tip[0] - index_tip[0]) / palm_scale
    index_middle_gap = abs(middle_tip[0] - index_tip[0]) / palm_scale
    index_dx = index_tip[0] - index_mcp[0]
    index_dy = index_tip[1] - index_mcp[1]
    index_is_vertical = abs(index_dy) > abs(index_dx) * 1.1

    # YES  — fist (all closed)
    if n_open == 0:
        return "yes", 0.92

    # ALL OPEN -> hello or stop (motion separates both)
    if n_open == 5:
        if motion > MOTION_HELLO_MIN:
            return "hello", 0.90
        return "stop", 0.88

    # ONLY INDEX + thumb folded -> help
    if _all_closed_except(fo, {"index"}):
        return "help", 0.87

    # ONLY INDEX + thumb open -> did_you_eat
    if _all_closed_except(fo, {"thumb", "index"}):
        if index_is_vertical and thumb_index_gap >= DID_YOU_EAT_THUMB_GAP_MIN:
            return "did_you_eat", 0.84
        if (not index_is_vertical) and pinch_ratio >= PINCH_PAIN_MIN:
            return "pain", 0.84
        return "pain", 0.72

    # INDEX + MIDDLE (peace/V) → no
    if _all_closed_except(fo, {"index", "middle"}) and index_middle_gap >= PEACE_GAP_MIN:
        return "no", 0.89

    # THUMB + PINKY (shaka) → call
    if _all_closed_except(fo, {"thumb", "pinky"}):
        return "call", 0.86

    # PINKY ONLY → water (ASL W approximation)
    if _all_closed_except(fo, {"pinky"}):
        return "water", 0.80

    # 3 fingers (index+middle+ring) → doctor
    if _all_closed_except(fo, {"index", "middle", "ring"}):
        return "doctor", 0.81

    # 4 fingers (no thumb) → bathroom
    if _all_closed_except(fo, {"index", "middle", "ring", "pinky"}):
        return "bathroom", 0.83

    # Thumb + index + middle → thanks (ASL flat hand near chin)
    if _all_closed_except(fo, {"thumb", "index", "middle"}):
        return "thanks", 0.79

    # More robust fallback than forcing one label.
    if fo["index"] and not fo["middle"] and not fo["ring"] and not fo["pinky"]:
        return "help", 0.58
    return "stop", 0.52


# ── public API ────────────────────────────────────────────────────────────────

def classify_landmarks(landmarks, motion=0.0):
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
    return _rule_classify(landmarks, motion=motion)
