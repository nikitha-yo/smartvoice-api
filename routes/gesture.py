from flask import Blueprint, request, jsonify
from ml.classifier import classify_landmarks
from models.database import get_conn
import traceback

gesture_bp = Blueprint("gesture", __name__)

GESTURE_SENTENCES = {
    "hello":       "Hello! How are you today?",
    "yes":         "Yes.",
    "no":          "No.",
    "help":        "Please help me. I need assistance.",
    "stop":        "Please stop.",
    "water":       "I need water, please.",
    "pain":        "I am in pain. Please help me.",
    "call":        "Please call someone for me.",
    "doctor":      "Please call a doctor for me.",
    "bathroom":    "I need to use the bathroom.",
    "thanks":      "Thank you.",
    "did_you_eat": "Did you eat?",
}


@gesture_bp.route("/predict", methods=["POST"])
def predict():
    """Receive 21 hand landmarks from the frontend and return gesture + sentence."""
    try:
        data = request.get_json(force=True)
        landmarks = data.get("landmarks", [])
        language  = data.get("language", "en")

        if not landmarks or len(landmarks) != 21:
            return jsonify({"error": "Expected 21 landmarks"}), 400

        motion = float(data.get("motion", 0.0) or 0.0)
        gesture, confidence = classify_landmarks(landmarks, motion=motion)
        sentence = GESTURE_SENTENCES.get(gesture, f"I want to say: {gesture}")

        conn = get_conn()
        conn.execute(
            "INSERT INTO gesture_history (gesture, sentence, language) VALUES (?, ?, ?)",
            (gesture, sentence, language),
        )
        conn.commit()
        conn.close()

        return jsonify({
            "gesture":    gesture,
            "sentence":   sentence,
            "confidence": round(confidence * 100, 1),
            "language":   language,
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@gesture_bp.route("/phrases", methods=["GET"])
def get_phrases():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM saved_phrases ORDER BY category, label").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@gesture_bp.route("/phrases", methods=["POST"])
def add_phrase():
    data = request.get_json(force=True)
    conn = get_conn()
    conn.execute(
        "INSERT INTO saved_phrases (label, sentence, category, language) VALUES (?,?,?,?)",
        (data["label"], data["sentence"], data.get("category", "custom"), data.get("language", "en")),
    )
    conn.commit()
    conn.close()
    return jsonify({"ok": True}), 201
