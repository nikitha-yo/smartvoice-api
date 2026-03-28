from flask import Blueprint, request, send_file, jsonify
from gtts import gTTS
import io, traceback

tts_bp = Blueprint("tts", __name__)

LANG_MAP = {
    "en": "en",
    "hi": "hi",
    "ta": "ta",
    "kn": "kn",
    "te": "te",
    "mr": "mr",
}


@tts_bp.route("/speak", methods=["POST"])
def speak():
    """Convert text to MP3 audio and stream it back."""
    try:
        data     = request.get_json(force=True)
        text     = data.get("text", "").strip()
        language = data.get("language", "en")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        lang_code = LANG_MAP.get(language, "en")
        tts = gTTS(text=text, lang=lang_code, slow=False)

        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)

        return send_file(buf, mimetype="audio/mpeg", as_attachment=False,
                         download_name="speech.mp3")

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500
