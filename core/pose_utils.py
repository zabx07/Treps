from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mediapipe as mp

from .geometry import distance, safe_ratio


POSE_LANDMARK = mp.solutions.pose.PoseLandmark

LANDMARK_INDEX = {
    "left_shoulder": POSE_LANDMARK.LEFT_SHOULDER.value,
    "right_shoulder": POSE_LANDMARK.RIGHT_SHOULDER.value,
    "left_elbow": POSE_LANDMARK.LEFT_ELBOW.value,
    "right_elbow": POSE_LANDMARK.RIGHT_ELBOW.value,
    "left_wrist": POSE_LANDMARK.LEFT_WRIST.value,
    "right_wrist": POSE_LANDMARK.RIGHT_WRIST.value,
    "left_hip": POSE_LANDMARK.LEFT_HIP.value,
    "right_hip": POSE_LANDMARK.RIGHT_HIP.value,
    "left_knee": POSE_LANDMARK.LEFT_KNEE.value,
    "right_knee": POSE_LANDMARK.RIGHT_KNEE.value,
    "left_ankle": POSE_LANDMARK.LEFT_ANKLE.value,
    "right_ankle": POSE_LANDMARK.RIGHT_ANKLE.value,
    "left_ear": POSE_LANDMARK.LEFT_EAR.value,
    "right_ear": POSE_LANDMARK.RIGHT_EAR.value,
    "left_foot_index": POSE_LANDMARK.LEFT_FOOT_INDEX.value,
    "right_foot_index": POSE_LANDMARK.RIGHT_FOOT_INDEX.value,
}


@dataclass
class SideLandmarks:
    side: str
    visibility: float
    points: dict[str, object]


def _prefixed_point(landmarks, side: str, point_name: str):
    return landmarks[LANDMARK_INDEX[f"{side}_{point_name}"]]


def _average_visibility(points: Iterable[object]) -> float:
    points = list(points)
    if not points:
        return 0.0
    return sum(max(getattr(point, "visibility", 0.0), getattr(point, "presence", 0.0)) for point in points) / len(points)


def point_visibility(point: object) -> float:
    return max(getattr(point, "visibility", 0.0), getattr(point, "presence", 0.0))


def select_body_side(
    landmarks,
    required_point_names: list[str],
    min_visibility: float,
    optional_point_names: list[str] | None = None,
    preferred_side: str | None = None,
    side_switch_margin: float = 0.08,
) -> SideLandmarks | None:
    optional_point_names = optional_point_names or []
    candidates: list[SideLandmarks] = []
    for side in ("left", "right"):
        required_points = {
            name: _prefixed_point(landmarks, side, name) for name in required_point_names
        }
        optional_points = {
            name: _prefixed_point(landmarks, side, name) for name in optional_point_names
        }
        visibility = _average_visibility(required_points.values())
        candidates.append(
            SideLandmarks(
                side=side,
                visibility=visibility,
                points={**required_points, **optional_points},
            )
        )

    candidates_by_side = {candidate.side: candidate for candidate in candidates}
    best = max(candidates, key=lambda candidate: candidate.visibility)
    if best.visibility < min_visibility:
        return None

    if preferred_side in candidates_by_side:
        preferred = candidates_by_side[preferred_side]
        visibility_gap = best.visibility - preferred.visibility
        if preferred.visibility >= min_visibility and visibility_gap <= side_switch_margin:
            return preferred
    return best


def is_side_view(landmarks, max_lateral_ratio: float) -> tuple[bool, float]:
    left_shoulder = landmarks[LANDMARK_INDEX["left_shoulder"]]
    right_shoulder = landmarks[LANDMARK_INDEX["right_shoulder"]]
    left_hip = landmarks[LANDMARK_INDEX["left_hip"]]
    right_hip = landmarks[LANDMARK_INDEX["right_hip"]]

    shoulder_center = (
        (left_shoulder.x + right_shoulder.x) / 2.0,
        (left_shoulder.y + right_shoulder.y) / 2.0,
    )
    hip_center = (
        (left_hip.x + right_hip.x) / 2.0,
        (left_hip.y + right_hip.y) / 2.0,
    )

    torso_size = max(distance(shoulder_center, hip_center), 1e-6)
    shoulder_span = abs(left_shoulder.x - right_shoulder.x)
    hip_span = abs(left_hip.x - right_hip.x)
    lateral_ratio = max(
        safe_ratio(shoulder_span, torso_size, default=1.0),
        safe_ratio(hip_span, torso_size, default=1.0),
    )
    return lateral_ratio <= max_lateral_ratio, lateral_ratio


def estimate_torso_size_ratio(landmarks) -> float:
    left_shoulder = landmarks[LANDMARK_INDEX["left_shoulder"]]
    right_shoulder = landmarks[LANDMARK_INDEX["right_shoulder"]]
    left_hip = landmarks[LANDMARK_INDEX["left_hip"]]
    right_hip = landmarks[LANDMARK_INDEX["right_hip"]]

    shoulder_center = (
        (left_shoulder.x + right_shoulder.x) / 2.0,
        (left_shoulder.y + right_shoulder.y) / 2.0,
    )
    hip_center = (
        (left_hip.x + right_hip.x) / 2.0,
        (left_hip.y + right_hip.y) / 2.0,
    )
    return distance(shoulder_center, hip_center)
