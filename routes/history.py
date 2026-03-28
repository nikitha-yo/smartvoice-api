from flask import Blueprint, jsonify, request
from models.database import get_conn

history_bp = Blueprint("history", __name__)


@history_bp.route("/", methods=["GET"])
def get_history():
    limit = int(request.args.get("limit", 50))
    conn  = get_conn()
    rows  = conn.execute(
        "SELECT * FROM gesture_history ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@history_bp.route("/clear", methods=["DELETE"])
def clear_history():
    conn = get_conn()
    conn.execute("DELETE FROM gesture_history")
    conn.commit()
    conn.close()
    return jsonify({"ok": True})
