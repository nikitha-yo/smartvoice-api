from flask import Flask
from flask_cors import CORS
from routes.gesture import gesture_bp
from routes.tts import tts_bp
from routes.history import history_bp
from models.database import init_db

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

app.register_blueprint(gesture_bp, url_prefix="/api/gesture")
app.register_blueprint(tts_bp,     url_prefix="/api/tts")
app.register_blueprint(history_bp, url_prefix="/api/history")

init_db()

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
