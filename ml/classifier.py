import numpy as np
from collections import deque, Counter

# ─── Landmark Indices ────────────────────────────────────────────────────────
WRIST       = 0
THUMB_MCP   = 2
THUMB_TIP   = 4
INDEX_MCP   = 5
INDEX_PIP   = 6
INDEX_TIP   = 8
MIDDLE_MCP  = 9
MIDDLE_PIP  = 10
MIDDLE_TIP  = 12
RING_MCP    = 13
RING_TIP    = 16
PINKY_MCP   = 17
PINKY_TIP   = 20

# ─── Tunable Thresholds ──────────────────────────────────────────────────────
THUMB_OPEN_DIST       = 0.20
THUMB_UP_MARGIN       = 0.12
WAVE_DIR_CHANGES      = 2
STATIC_STD_THRESHOLD  = 0.012
PAIN_ANGLE_MAX        = 55
EAT_ANGLE_MIN         = 62
CIRCLE_X_MIN          = 0.045
CIRCLE_Y_MIN          = 0.035
HUNGRY_RISE_MIN       = 0.035
SMOOTH_FRAMES         = 5
SMOOTH_THRESHOLD      = 3

gesture_buffer   = deque(maxlen=SMOOTH_FRAMES)
position_history = deque(maxlen=12)


def _to_array(landmarks):
    rows = []
    for p in landmarks:
        if isinstance(p, dict):
            rows.append([float(p["x"]), float(p["y"]), float(p.get("z", 0.0))])
        else:
            rows.append([float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0])
    return np.array(rows, dtype=np.float32)


def normalize(lms):
    mn = lms[:, :2].min(axis=0)
    mx = lms[:, :2].max(axis=0)
    box = np.linalg.norm(mx - mn) + 1e-6
    out = lms.copy()
    out[:, :2] = (lms[:, :2] - mn) / box
    return out


def dist(lms, a, b):
    return float(np.linalg.norm(lms[a][:2] - lms[b][:2]))


def angle_between(lms, tip_a, mcp_a, tip_b, mcp_b):
    v1 = lms[tip_a][:2] - lms[mcp_a][:2]
    v2 = lms[tip_b][:2] - lms[mcp_b][:2]
    n1 = np.linalg.norm(v1) + 1e-6
    n2 = np.linalg.norm(v2) + 1e-6
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def finger_open(lms, tip, mcp, pip=None):
    tip_above_mcp = lms[tip][1] < lms[mcp][1]
    if pip is not None:
        tip_above_pip = lms[tip][1] < lms[pip][1]
        return tip_above_mcp and tip_above_pip
    return tip_above_mcp


def get_finger_states(lms):
    return {
        "T": dist(lms, THUMB_TIP, THUMB_MCP) > THUMB_OPEN_DIST,
        "I": finger_open(lms, INDEX_TIP, INDEX_MCP, INDEX_PIP),
        "M": finger_open(lms, MIDDLE_TIP, MIDDLE_MCP, MIDDLE_PIP),
        "R": finger_open(lms, RING_TIP, RING_MCP),
        "P": finger_open(lms, PINKY_TIP, PINKY_MCP),
    }


def thumb_pointing_up(lms):
    tip_y = lms[THUMB_TIP][1]
    mcp_y = lms[THUMB_MCP][1]
    wrist_y = lms[WRIST][1]
    return (mcp_y - tip_y) > THUMB_UP_MARGIN and tip_y < wrist_y


def all_truly_closed(lms):
    checks = [
        lms[INDEX_TIP][1] >= lms[INDEX_PIP][1],
        lms[MIDDLE_TIP][1] >= lms[MIDDLE_PIP][1],
        lms[RING_TIP][1] >= lms[RING_MCP][1],
        lms[PINKY_TIP][1] >= lms[PINKY_MCP][1],
    ]
    return all(checks)


def is_static(history):
    if len(history) < 4:
        return False
    positions = np.array(list(history)[-6:])
    return float(np.std(positions, axis=0).max()) < STATIC_STD_THRESHOLD


def has_wave(history):
    if len(history) < 6:
        return False
    xs = [p[0] for p in history]
    travel = max(xs) - min(xs)
    if travel < 0.06:
        return False
    changes = sum(
        1 for i in range(1, len(xs) - 1)
        if (xs[i] - xs[i - 1]) * (xs[i + 1] - xs[i]) < -0.001
    )
    return changes >= WAVE_DIR_CHANGES


def has_upward_motion(history):
    if len(history) < 6:
        return False
    pts = list(history)
    mid = len(pts) // 2
    avg_early = np.mean([p[1] for p in pts[:mid]])
    avg_late = np.mean([p[1] for p in pts[mid:]])
    return (avg_early - avg_late) > HUNGRY_RISE_MIN


def has_circular_motion(history):
    if len(history) < 8:
        return False
    pts = np.array(list(history))
    x_range = float(pts[:, 0].max() - pts[:, 0].min())
    y_range = float(pts[:, 1].max() - pts[:, 1].min())
    return x_range > CIRCLE_X_MIN and y_range > CIRCLE_Y_MIN


def check_sorry(lms, fs, history):
    if not any(fs.values()) and all_truly_closed(lms) and not thumb_pointing_up(lms):
        if has_circular_motion(history):
            return ("sorry", 0.90)
    return None


def check_hello(lms, fs, history):
    if all(fs.values()) and has_wave(history):
        return ("hello", 0.95)
    return None


def check_hungry(lms, fs, history):
    if all(fs.values()) and has_upward_motion(history):
        return ("hungry", 0.90)
    return None


def check_stop(lms, fs, history):
    if all(fs.values()) and is_static(history):
        return ("stop", 0.93)
    return None


def check_good(lms, fs):
    if not fs["I"] and not fs["M"] and not fs["R"] and not fs["P"]:
        if thumb_pointing_up(lms) and all_truly_closed(lms):
            return ("good", 0.95)
    return None


def check_yes(lms, fs):
    if not any(fs.values()) and all_truly_closed(lms) and not thumb_pointing_up(lms):
        return ("yes", 0.95)
    return None


def check_help(lms, fs):
    if fs["I"] and not fs["M"] and not fs["R"] and not fs["P"] and not fs["T"]:
        if (lms[INDEX_MCP][1] - lms[INDEX_TIP][1]) > 0.08:
            return ("help", 0.95)
    return None


def check_water(lms, fs):
    if fs["P"] and not fs["I"] and not fs["M"] and not fs["R"] and not fs["T"]:
        if (lms[PINKY_MCP][1] - lms[PINKY_TIP][1]) > 0.06:
            return ("water", 0.95)
    return None


def check_call(lms, fs):
    if fs["T"] and fs["P"] and not fs["I"] and not fs["M"] and not fs["R"]:
        if lms[INDEX_TIP][1] >= lms[INDEX_PIP][1] and lms[MIDDLE_TIP][1] >= lms[MIDDLE_PIP][1]:
            return ("call", 0.95)
    return None


def check_pain_and_eat(lms, fs):
    if fs["T"] and fs["I"] and not fs["M"] and not fs["R"] and not fs["P"]:
        if lms[MIDDLE_TIP][1] >= lms[MIDDLE_PIP][1]:
            ang = angle_between(lms, INDEX_TIP, INDEX_MCP, THUMB_TIP, THUMB_MCP)
            if ang <= PAIN_ANGLE_MAX:
                return ("pain", 0.92)
            if ang >= EAT_ANGLE_MIN:
                return ("did_you_eat", 0.90)
    return None


def check_doctor(lms, fs):
    if fs["I"] and fs["M"] and fs["R"] and not fs["T"] and not fs["P"]:
        if lms[PINKY_TIP][1] >= lms[PINKY_MCP][1]:
            return ("doctor", 0.95)
    return None


def check_bathroom(lms, fs):
    if fs["I"] and fs["M"] and fs["R"] and fs["P"] and not fs["T"]:
        if dist(lms, THUMB_TIP, THUMB_MCP) < 0.22:
            return ("bathroom", 0.95)
    return None


def check_thanks(lms, fs):
    if fs["T"] and fs["I"] and fs["M"] and not fs["R"] and not fs["P"]:
        if lms[RING_TIP][1] >= lms[RING_MCP][1]:
            return ("thanks", 0.93)
    return None


def check_no(lms, fs):
    if fs["I"] and fs["M"] and not fs["T"] and not fs["R"] and not fs["P"]:
        return ("no", 0.93)
    return None


def classify_landmarks(landmarks, motion=0.0):
    # motion param kept for API compatibility; classifier uses wrist history itself.
    lms = normalize(_to_array(landmarks))
    fs = get_finger_states(lms)
    position_history.append(tuple(lms[WRIST][:2]))

    result = (
        check_sorry(lms, fs, position_history) or
        check_hello(lms, fs, position_history) or
        check_hungry(lms, fs, position_history) or
        check_stop(lms, fs, position_history) or
        check_good(lms, fs) or
        check_yes(lms, fs) or
        check_call(lms, fs) or
        check_bathroom(lms, fs) or
        check_doctor(lms, fs) or
        check_thanks(lms, fs) or
        check_pain_and_eat(lms, fs) or
        check_no(lms, fs) or
        check_water(lms, fs) or
        check_help(lms, fs)
    )

    raw = result[0] if result else "unknown"
    gesture_buffer.append(raw)
    counts = Counter(gesture_buffer)
    best, freq = counts.most_common(1)[0]
    final = best if freq >= SMOOTH_THRESHOLD and best != "unknown" else "unknown"
    conf = result[1] if result and final != "unknown" else 0.0
    return final, float(conf)
