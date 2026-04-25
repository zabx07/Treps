from .evaluator import evaluate_model
from .video import evaluate_labeled_videos, load_manifest_video_specs, run_video_session

__all__ = [
    "ExerciseSession",
    "evaluate_labeled_videos",
    "evaluate_model",
    "load_manifest_video_specs",
    "run_video_session",
]


def __getattr__(name):
    if name == "ExerciseSession":
        from .engine import ExerciseSession

        return ExerciseSession
    raise AttributeError(f"module 'core' has no attribute {name!r}")
