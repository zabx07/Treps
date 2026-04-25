from __future__ import annotations

import tempfile
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

import config
from core.video import run_video_session


main_bp = Blueprint("main", __name__)
EXERCISE_TYPES = set(config.EXERCISE_RUNTIME_CONFIGS)
ALLOWED_SERVER_UPLOAD_EXTENSIONS = {
    extension.lower() for extension in config.SERVER_UPLOAD_FALLBACK_ALLOWED_EXTENSIONS
}


def _allowed_server_upload(filename: str) -> bool:
    suffix = Path(filename).suffix.lower().lstrip(".")
    return bool(suffix and suffix in ALLOWED_SERVER_UPLOAD_EXTENSIONS)


def _safe_target_reps(value) -> int:
    try:
        target_reps = int(value)
        return target_reps if target_reps > 0 else 10
    except (TypeError, ValueError):
        return 10


def _safe_optional_target_reps(value) -> int | None:
    try:
        if value in {"", None, "null", "None"}:
            return None
        target_reps = int(value)
        return target_reps if target_reps > 0 else None
    except (TypeError, ValueError):
        return None


def _summarize_history(history: list[dict]) -> tuple[int, int, str]:
    good_reps = sum(1 for rep in history if rep.get("status") == "benar")
    bad_reps = sum(1 for rep in history if rep.get("status") == "salah")
    if not history:
        return good_reps, bad_reps, "Belum ada repetisi terdeteksi."
    if bad_reps == 0:
        return good_reps, bad_reps, "Teknik stabil pada sesi ini."
    return good_reps, bad_reps, f"{bad_reps} repetisi perlu koreksi dari {len(history)} repetisi."


def _dominant_warning(history: list[dict], runtime_result: dict) -> str:
    warning_counts = Counter(
        rep.get("detail") for rep in history if rep.get("status") == "salah" and rep.get("detail")
    )
    if warning_counts:
        return warning_counts.most_common(1)[0][0]
    if runtime_result.get("invalid_view_rate", 0) > 0:
        return "Kamera belum stabil dari samping."
    if runtime_result.get("low_visibility_rate", 0) > 0 or runtime_result.get("no_pose_rate", 0) > 0:
        return "Landmark belum selalu terbaca dengan baik."
    return "Tidak ada warning dominan."


def _estimate_video_duration_seconds(video_path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    capture.release()
    if frame_count <= 0 or fps <= 0:
        return 0
    return max(0, round(frame_count / fps))


def _build_backend_session_summary(
    *,
    exercise_type: str,
    target_reps: int | None,
    file_name: str,
    runtime_result: dict,
) -> dict:
    history = runtime_result.get("history", [])
    rep_count = runtime_result.get("rep_count", 0)
    good_reps, bad_reps, technique_summary = _summarize_history(history)
    duration_seconds = _estimate_video_duration_seconds(Path(runtime_result["video_path"]))
    return {
        "id": str(uuid.uuid4()),
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "reason": "backend_upload_fallback",
        "exerciseType": exercise_type,
        "targetReps": target_reps,
        "repCount": rep_count,
        "goodReps": good_reps,
        "badReps": bad_reps,
        "durationSeconds": duration_seconds,
        "mode": "upload",
        "modeLabel": "Upload video",
        "runtimePreset": "backend_fallback",
        "techniqueSummary": technique_summary,
        "dominantWarning": _dominant_warning(history, runtime_result),
        "metrics": {
            "noPoseRate": runtime_result.get("no_pose_rate", 0.0),
            "lowVisibilityRate": runtime_result.get("low_visibility_rate", 0.0),
            "invalidViewRate": runtime_result.get("invalid_view_rate", 0.0),
            "tooFarRate": runtime_result.get("too_far_rate", 0.0),
            "inferenceErrors": 0,
            "processingSeconds": runtime_result.get("processing_seconds", 0.0),
        },
        "repHistory": history[-30:],
        "sourceFileName": file_name,
        "backendFallback": True,
    }


@main_bp.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response


@main_bp.route("/options", methods=["OPTIONS"])
def handle_options():
    return "", 200


@main_bp.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "runtime_mode": "client_camera_plus_optional_batch_upload_fallback",
            "upload_backend_fallback_enabled": config.SERVER_UPLOAD_FALLBACK_ENABLED,
        }
    )


@main_bp.route("/")
def index():
    return render_template("home.html")


@main_bp.route("/tutorial")
def tutorial_index():
    return render_template("tutorial.html", exercise_type=None)


@main_bp.route("/exercise/<exercise_type>")
def exercise(exercise_type):
    if exercise_type not in EXERCISE_TYPES:
        return "Not found", 404
    return render_template(
        "exercise.html",
        exercise_type=exercise_type,
        runtime_config=config.build_web_runtime_payload(exercise_type),
    )


@main_bp.route("/tutorial/<exercise_type>")
def tutorial(exercise_type):
    if exercise_type not in EXERCISE_TYPES:
        return "Not found", 404
    return render_template("tutorial.html", exercise_type=exercise_type)


@main_bp.route("/about")
def about():
    return render_template("about.html")


@main_bp.route("/api/start_session", methods=["POST"])
def start_session():
    try:
        data = request.get_json(silent=True) or {}
        exercise_type = data.get("exercise_type", "squat")
        if exercise_type not in EXERCISE_TYPES:
            return jsonify({"error": f"Invalid exercise type: {exercise_type}"}), 400

        target_reps = _safe_target_reps(data.get("target_reps", 10))
        return jsonify(
            {
                "session_id": str(uuid.uuid4()),
                "exercise_type": exercise_type,
                "target_reps": target_reps,
                "runtime_mode": "client",
                "upload_backend_fallback_enabled": config.SERVER_UPLOAD_FALLBACK_ENABLED,
                "upload_backend_fallback_route": config.SERVER_UPLOAD_FALLBACK_ROUTE,
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Error: {exc}"}), 500


@main_bp.route(config.SERVER_UPLOAD_FALLBACK_ROUTE, methods=["POST"])
def process_uploaded_video():
    if not config.SERVER_UPLOAD_FALLBACK_ENABLED:
        return jsonify({"error": "Upload fallback backend dinonaktifkan."}), 503

    uploaded_file = request.files.get("video")
    if uploaded_file is None:
        return jsonify({"error": "File video tidak ditemukan pada request."}), 400
    if not uploaded_file.filename:
        return jsonify({"error": "Nama file video kosong."}), 400
    if not _allowed_server_upload(uploaded_file.filename):
        allowed = ", ".join(sorted(ALLOWED_SERVER_UPLOAD_EXTENSIONS))
        return jsonify(
            {"error": f"Format video tidak didukung untuk fallback backend. Gunakan: {allowed}."}
        ), 400

    exercise_type = request.form.get("exercise_type", "squat")
    if exercise_type not in EXERCISE_TYPES:
        return jsonify({"error": f"Invalid exercise type: {exercise_type}"}), 400

    target_reps = _safe_optional_target_reps(request.form.get("target_reps"))
    temp_path = None
    try:
        suffix = Path(secure_filename(uploaded_file.filename)).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            uploaded_file.save(handle)
            temp_path = Path(handle.name)

        runtime_result = run_video_session(
            temp_path,
            exercise_type=exercise_type,
            max_reps=None,
            draw_pose=False,
        )
        session_summary = _build_backend_session_summary(
            exercise_type=exercise_type,
            target_reps=target_reps,
            file_name=uploaded_file.filename,
            runtime_result=runtime_result,
        )
        return jsonify(
            {
                "mode": "backend_upload_fallback",
                "message": "Video berhasil diproses dengan batch fallback backend.",
                "session_summary": session_summary,
            }
        )
    except Exception as exc:
        return jsonify(
            {
                "error": (
                    "Batch fallback backend gagal memproses video. "
                    "Gunakan MP4/MOV yang lebih pendek atau cek dependency OpenCV/MediaPipe."
                ),
                "detail": str(exc),
            }
        ), 500
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                current_app.logger.warning("Temporary upload file could not be removed: %s", temp_path)


@main_bp.route("/api/cleanup", methods=["POST"])
def cleanup():
    return jsonify({"status": "cleaned"})


@main_bp.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Not found"}), 404


@main_bp.errorhandler(413)
@main_bp.app_errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    return (
        jsonify(
            {
                "error": (
                    f"File terlalu besar untuk backend fallback. "
                    f"Batas server: {config.SERVER_UPLOAD_FALLBACK_MAX_FILE_MB} MB."
                )
            }
        ),
        413,
    )


@main_bp.errorhandler(500)
def server_error(_error):
    return jsonify({"error": "Internal server error"}), 500
