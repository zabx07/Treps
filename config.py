"""Central configuration for shared pose-based exercise analysis."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _read_dxdiag_text() -> str:
    for file_name in ("dxdiag.txt", "DxDiag.txt"):
        dxdiag_path = PROJECT_ROOT / file_name
        if dxdiag_path.exists():
            return dxdiag_path.read_text(encoding="utf-8", errors="ignore")
    return ""


def _extract_first(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_int(pattern: str, text: str) -> int | None:
    raw_value = _extract_first(pattern, text)
    digits = re.sub(r"[^\d]", "", raw_value)
    return int(digits) if digits else None


def _build_device_runtime_profile() -> dict:
    dxdiag_text = _read_dxdiag_text()
    if not dxdiag_text:
        return {
            "dxdiag_available": False,
            "machine_name": "",
            "processor": "",
            "memory_mb": None,
            "integrated_gpu": "",
            "discrete_gpu": "",
            "quality_preset": "balanced_default",
            "camera_width": 960,
            "camera_height": 540,
            "camera_frame_rate": 24,
            "process_interval_ms": 75,
            "upload_max_file_mb": 200,
            "upload_max_duration_seconds": 120,
            "upload_downscale_max_dimension": 960,
            "upload_frame_step_ms": 100,
            "upload_seek_timeout_ms": 4000,
            "upload_error_retry_limit": 3,
            "upload_fallback_dimensions": [960, 720, 640],
        }

    memory_mb = _extract_int(r"Memory:\s*([0-9,]+)\s*MB RAM", dxdiag_text)
    processor = _extract_first(r"Processor:\s*(.+)", dxdiag_text)
    machine_name = _extract_first(r"Machine name:\s*(.+)", dxdiag_text)
    integrated_gpu = _extract_first(r"Card name:\s*(Intel[^\r\n]+)", dxdiag_text)
    discrete_gpu = _extract_first(r"Card name:\s*(NVIDIA[^\r\n]+)", dxdiag_text)

    high_midrange_device = bool(
        memory_mb and memory_mb >= 16000 and ("12500H" in processor or "RTX 3050" in discrete_gpu)
    )
    if high_midrange_device:
        return {
            "dxdiag_available": True,
            "machine_name": machine_name,
            "processor": processor,
            "memory_mb": memory_mb,
            "integrated_gpu": integrated_gpu,
            "discrete_gpu": discrete_gpu,
            "quality_preset": "balanced_plus",
            "camera_width": 960,
            "camera_height": 540,
            "camera_frame_rate": 30,
            "process_interval_ms": 67,
            "upload_max_file_mb": 300,
            "upload_max_duration_seconds": 180,
            "upload_downscale_max_dimension": 960,
            "upload_frame_step_ms": 90,
            "upload_seek_timeout_ms": 4000,
            "upload_error_retry_limit": 3,
            "upload_fallback_dimensions": [960, 720, 640],
        }

    return {
        "dxdiag_available": True,
        "machine_name": machine_name,
        "processor": processor,
        "memory_mb": memory_mb,
        "integrated_gpu": integrated_gpu,
        "discrete_gpu": discrete_gpu,
        "quality_preset": "balanced_safe",
        "camera_width": 848,
        "camera_height": 480,
        "camera_frame_rate": 24,
        "process_interval_ms": 84,
        "upload_max_file_mb": 200,
        "upload_max_duration_seconds": 120,
        "upload_downscale_max_dimension": 720,
        "upload_frame_step_ms": 120,
        "upload_seek_timeout_ms": 5000,
        "upload_error_retry_limit": 4,
        "upload_fallback_dimensions": [720, 640, 512],
    }


DEVICE_RUNTIME_PROFILE = _build_device_runtime_profile()


# ================= GLOBAL =================
POSE_MIN_DETECTION_CONFIDENCE = 0.6
POSE_MIN_TRACKING_CONFIDENCE = 0.6
POSE_MIN_PRESENCE_CONFIDENCE = 0.5
LANDMARK_VISIBILITY_THRESHOLD = 0.55
LANDMARK_MIN_POINT_VISIBILITY_THRESHOLD = 0.35
ANGLE_SMOOTHING_WINDOW = 6
LATERAL_RATIO_SMOOTHING_WINDOW = 4
STATE_STABLE_FRAMES = 3
MIN_REP_WINDOW_FRAMES = 6
FORM_VIOLATION_RATIO_THRESHOLD = 0.25
SIDE_VIEW_REQUIRED = True
SIDE_VIEW_MAX_LATERAL_RATIO = 0.45
SIDE_SWITCH_VISIBILITY_MARGIN = 0.08
MIN_TORSO_SIZE_RATIO = 0.14
READY_POSITION_REQUIRED = True

WEB_RUNTIME_WASM_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
WEB_RUNTIME_MODEL_ASSET_PATH = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
WEB_RUNTIME_NUM_POSES = 1
WEB_RUNTIME_PROCESS_INTERVAL_MS = DEVICE_RUNTIME_PROFILE["process_interval_ms"]
WEB_RUNTIME_CAMERA_WIDTH = DEVICE_RUNTIME_PROFILE["camera_width"]
WEB_RUNTIME_CAMERA_HEIGHT = DEVICE_RUNTIME_PROFILE["camera_height"]
WEB_RUNTIME_CAMERA_FRAME_RATE = DEVICE_RUNTIME_PROFILE["camera_frame_rate"]
WEB_RUNTIME_UPLOAD_MAX_FILE_MB = DEVICE_RUNTIME_PROFILE["upload_max_file_mb"]
WEB_RUNTIME_UPLOAD_MAX_DURATION_SECONDS = DEVICE_RUNTIME_PROFILE["upload_max_duration_seconds"]
WEB_RUNTIME_UPLOAD_DOWNSCALE_MAX_DIMENSION = DEVICE_RUNTIME_PROFILE["upload_downscale_max_dimension"]
WEB_RUNTIME_UPLOAD_FRAME_STEP_MS = DEVICE_RUNTIME_PROFILE["upload_frame_step_ms"]
WEB_RUNTIME_UPLOAD_SEEK_TIMEOUT_MS = DEVICE_RUNTIME_PROFILE["upload_seek_timeout_ms"]
WEB_RUNTIME_UPLOAD_ERROR_RETRY_LIMIT = DEVICE_RUNTIME_PROFILE["upload_error_retry_limit"]
WEB_RUNTIME_UPLOAD_FALLBACK_DIMENSIONS = DEVICE_RUNTIME_PROFILE["upload_fallback_dimensions"]
WEB_RUNTIME_QUALITY_PRESET = DEVICE_RUNTIME_PROFILE["quality_preset"]
SERVER_UPLOAD_FALLBACK_ENABLED = True
SERVER_UPLOAD_FALLBACK_MAX_FILE_MB = 512
SERVER_UPLOAD_FALLBACK_ALLOWED_EXTENSIONS = ["mp4", "mov", "m4v", "webm", "avi", "mkv"]
SERVER_UPLOAD_FALLBACK_ROUTE = "/api/process_uploaded_video"

ARTIFACTS_DIR = "artifacts"
DATASET_MANIFEST_PATH = "artifacts/dataset_manifest.csv"
DATASET_ISSUES_PATH = "artifacts/dataset_issues.json"
EVALUATIONS_DIR = "artifacts/evaluations"
MANIFEST_REQUIRED_FIELDS = [
    "file_path",
    "file_name",
    "exercise_type",
    "label_main",
    "error_tags",
    "expected_reps",
    "subject_id",
    "camera_view",
    "occlusion_level",
    "clip_source",
    "source_video",
    "split",
    "notes",
    "split_group",
    "label_group",
    "size_bytes",
    "sha256",
]
MANIFEST_MANUAL_FIELDS = [
    "expected_reps",
    "subject_id",
    "camera_view",
    "occlusion_level",
    "split",
    "notes",
]

# ================= SQUAT =================
SQUAT_TOP_ANGLE = 165
SQUAT_BOTTOM_ANGLE = 95
SQUAT_TRACK_START_ANGLE = 145
SQUAT_ROM_THRESHOLD = 98
SQUAT_TORSO_LEAN_THRESHOLD = 58
SQUAT_KNEE_FORWARD_RATIO = None
SQUAT_HEEL_LIFT_RATIO_THRESHOLD = 10.0

# ================= PUSH-UP =================
PUSHUP_TOP_ANGLE = 165
PUSHUP_BOTTOM_ANGLE = 95
PUSHUP_TRACK_START_ANGLE = 145
PUSHUP_ROM_THRESHOLD = 90
PUSHUP_BODY_ALIGNMENT_THRESHOLD = 160
PUSHUP_TRAPS_RAISE_RATIO_THRESHOLD = 2.5
PUSHUP_TRAPS_MIN_ELBOW_ANGLE = 70

# ================= FEEDBACK =================
MISSING_POSE_FEEDBACK = "Pose tidak terdeteksi. Pastikan seluruh tubuh terlihat."
LOW_VISIBILITY_FEEDBACK = "Landmark tubuh utama belum terlihat jelas."
SIDE_VIEW_FEEDBACK = "Posisikan tubuh menyamping ke kamera."
TOO_FAR_FEEDBACK = "Dekatkan tubuh ke kamera agar evaluasi lebih stabil."
UPLOAD_SIZE_FEEDBACK = "File terlalu besar. Gunakan video lebih pendek atau resolusi lebih rendah."
UPLOAD_FORMAT_FEEDBACK = "Format video kurang stabil. Gunakan MP4, MOV, M4V, atau WebM."
UPLOAD_RUNTIME_FEEDBACK = "Video diproses dengan mode hemat agar inference browser lebih stabil."
UPLOAD_BACKEND_FALLBACK_FEEDBACK = "Treps beralih ke batch fallback di backend untuk file ini."


WEB_RUNTIME_GLOBAL_CONFIG = {
    "pose_min_detection_confidence": POSE_MIN_DETECTION_CONFIDENCE,
    "pose_min_tracking_confidence": POSE_MIN_TRACKING_CONFIDENCE,
    "pose_min_presence_confidence": POSE_MIN_PRESENCE_CONFIDENCE,
    "landmark_visibility_threshold": LANDMARK_VISIBILITY_THRESHOLD,
    "landmark_min_point_visibility_threshold": LANDMARK_MIN_POINT_VISIBILITY_THRESHOLD,
    "angle_smoothing_window": ANGLE_SMOOTHING_WINDOW,
    "lateral_ratio_smoothing_window": LATERAL_RATIO_SMOOTHING_WINDOW,
    "state_stable_frames": STATE_STABLE_FRAMES,
    "min_rep_window_frames": MIN_REP_WINDOW_FRAMES,
    "form_violation_ratio_threshold": FORM_VIOLATION_RATIO_THRESHOLD,
    "side_view_required": SIDE_VIEW_REQUIRED,
    "side_view_max_lateral_ratio": SIDE_VIEW_MAX_LATERAL_RATIO,
    "side_switch_visibility_margin": SIDE_SWITCH_VISIBILITY_MARGIN,
    "min_torso_size_ratio": MIN_TORSO_SIZE_RATIO,
    "ready_position_required": READY_POSITION_REQUIRED,
    "missing_pose_feedback": MISSING_POSE_FEEDBACK,
    "low_visibility_feedback": LOW_VISIBILITY_FEEDBACK,
    "side_view_feedback": SIDE_VIEW_FEEDBACK,
    "too_far_feedback": TOO_FAR_FEEDBACK,
    "upload_size_feedback": UPLOAD_SIZE_FEEDBACK,
    "upload_format_feedback": UPLOAD_FORMAT_FEEDBACK,
    "upload_runtime_feedback": UPLOAD_RUNTIME_FEEDBACK,
    "web_runtime_wasm_root": WEB_RUNTIME_WASM_ROOT,
    "web_runtime_model_asset_path": WEB_RUNTIME_MODEL_ASSET_PATH,
    "web_runtime_num_poses": WEB_RUNTIME_NUM_POSES,
    "web_runtime_process_interval_ms": WEB_RUNTIME_PROCESS_INTERVAL_MS,
    "camera_width": WEB_RUNTIME_CAMERA_WIDTH,
    "camera_height": WEB_RUNTIME_CAMERA_HEIGHT,
    "camera_frame_rate": WEB_RUNTIME_CAMERA_FRAME_RATE,
    "upload_max_file_mb": WEB_RUNTIME_UPLOAD_MAX_FILE_MB,
    "upload_max_duration_seconds": WEB_RUNTIME_UPLOAD_MAX_DURATION_SECONDS,
    "upload_downscale_max_dimension": WEB_RUNTIME_UPLOAD_DOWNSCALE_MAX_DIMENSION,
    "upload_frame_step_ms": WEB_RUNTIME_UPLOAD_FRAME_STEP_MS,
    "upload_seek_timeout_ms": WEB_RUNTIME_UPLOAD_SEEK_TIMEOUT_MS,
    "upload_error_retry_limit": WEB_RUNTIME_UPLOAD_ERROR_RETRY_LIMIT,
    "upload_fallback_dimensions": list(WEB_RUNTIME_UPLOAD_FALLBACK_DIMENSIONS),
    "upload_backend_fallback_enabled": SERVER_UPLOAD_FALLBACK_ENABLED,
    "upload_backend_fallback_max_file_mb": SERVER_UPLOAD_FALLBACK_MAX_FILE_MB,
    "upload_backend_fallback_route": SERVER_UPLOAD_FALLBACK_ROUTE,
    "upload_backend_fallback_extensions": list(SERVER_UPLOAD_FALLBACK_ALLOWED_EXTENSIONS),
    "runtime_quality_preset": WEB_RUNTIME_QUALITY_PRESET,
    "upload_backend_fallback_feedback": UPLOAD_BACKEND_FALLBACK_FEEDBACK,
    "device_profile": {
        "dxdiag_available": DEVICE_RUNTIME_PROFILE["dxdiag_available"],
        "machine_name": DEVICE_RUNTIME_PROFILE["machine_name"],
        "processor": DEVICE_RUNTIME_PROFILE["processor"],
        "memory_mb": DEVICE_RUNTIME_PROFILE["memory_mb"],
        "integrated_gpu": DEVICE_RUNTIME_PROFILE["integrated_gpu"],
        "discrete_gpu": DEVICE_RUNTIME_PROFILE["discrete_gpu"],
        "quality_preset": DEVICE_RUNTIME_PROFILE["quality_preset"],
    },
}


EXERCISE_RUNTIME_CONFIGS = {
    "push_up": {
        "exercise_type": "push_up",
        "label": "Push-up",
        "required_points": ["shoulder", "elbow", "wrist", "hip", "ankle"],
        "optional_points": ["ear"],
        "top_angle": PUSHUP_TOP_ANGLE,
        "bottom_angle": PUSHUP_BOTTOM_ANGLE,
        "track_start_angle": PUSHUP_TRACK_START_ANGLE,
        "rom_threshold": PUSHUP_ROM_THRESHOLD,
        "body_alignment_threshold": PUSHUP_BODY_ALIGNMENT_THRESHOLD,
        "head_drop_threshold": PUSHUP_TRAPS_RAISE_RATIO_THRESHOLD,
        "head_drop_min_main_angle_threshold": PUSHUP_TRAPS_MIN_ELBOW_ANGLE,
        "default_feedback": "Siapkan posisi plank dari samping kamera.",
        "ready_pose_feedback": "Mulai dari plank penuh agar repetisi dihitung dengan benar.",
        "stage_up_feedback": "Turunkan badan secara terkendali.",
        "stage_mid_feedback": "Jaga tubuh tetap lurus saat turun.",
        "stage_down_feedback": "Dorong kembali ke posisi plank.",
        "error_catalog": [
            {"code": "tidak_full_rom", "label": "Tidak full ROM"},
            {"code": "badan_bungkuk", "label": "Badan tidak lurus"},
            {"code": "traps_naik", "label": "Bahu/trapezius terlalu naik"},
        ],
    },
    "squat": {
        "exercise_type": "squat",
        "label": "Squat",
        "required_points": ["shoulder", "hip", "knee", "ankle"],
        "optional_points": ["heel", "foot_index"],
        "top_angle": SQUAT_TOP_ANGLE,
        "bottom_angle": SQUAT_BOTTOM_ANGLE,
        "track_start_angle": SQUAT_TRACK_START_ANGLE,
        "rom_threshold": SQUAT_ROM_THRESHOLD,
        "torso_lean_threshold": SQUAT_TORSO_LEAN_THRESHOLD,
        "knee_forward_ratio_threshold": SQUAT_KNEE_FORWARD_RATIO,
        "heel_lift_ratio_threshold": SQUAT_HEEL_LIFT_RATIO_THRESHOLD,
        "default_feedback": "Siapkan posisi berdiri menyamping kamera.",
        "ready_pose_feedback": "Mulai dari posisi berdiri tegak sebelum turun.",
        "stage_up_feedback": "Turunkan pinggul ke belakang.",
        "stage_mid_feedback": "Jaga dada terbuka dan lutut stabil.",
        "stage_down_feedback": "Dorong kembali ke posisi berdiri.",
        "error_catalog": [
            {"code": "tidak_full_rom", "label": "Depth kurang / tidak full ROM"},
            {"code": "badan_bungkuk", "label": "Badan terlalu condong ke depan"},
            {"code": "lutut_maju", "label": "Lutut terlalu maju"},
            {"code": "kaki_jinjit", "label": "Tumit terangkat / kaki jinjit"},
        ],
    },
}


def build_web_runtime_payload(exercise_type: str) -> dict:
    exercise_config = EXERCISE_RUNTIME_CONFIGS[exercise_type]
    return {
        "exercise_type": exercise_type,
        "global": dict(WEB_RUNTIME_GLOBAL_CONFIG),
        "exercise": {
            key: (list(value) if isinstance(value, list) else value)
            for key, value in exercise_config.items()
        },
    }


# ================= LEGACY ALIASES =================
KNEE_DOWN = SQUAT_BOTTOM_ANGLE
KNEE_UP = SQUAT_TOP_ANGLE
ROM_THRESHOLD = SQUAT_ROM_THRESHOLD
BACK_THRESHOLD = SQUAT_TORSO_LEAN_THRESHOLD
KNEE_FORWARD_RATIO = SQUAT_KNEE_FORWARD_RATIO
ELBOW_DOWN = PUSHUP_BOTTOM_ANGLE
ELBOW_UP = PUSHUP_TOP_ANGLE
BODY_ALIGNMENT_THRESHOLD = PUSHUP_BODY_ALIGNMENT_THRESHOLD
HEAD_DROP_THRESHOLD = PUSHUP_TRAPS_RAISE_RATIO_THRESHOLD
