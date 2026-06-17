from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import mediapipe as mp

import config
from .geometry import as_pixel, calculate_angle, calculate_vertical_lean, safe_ratio
from .pose_utils import SideLandmarks, estimate_torso_size_ratio, is_side_view, select_body_side


@dataclass(frozen=True)
class ExerciseConfig:
    exercise_type: str
    label: str
    required_points: list[str]
    optional_points: list[str]
    error_catalog: list[dict]
    default_feedback: str
    top_angle: float
    bottom_angle: float
    track_start_angle: float
    rom_threshold: float
    ready_pose_feedback: str
    stage_up_feedback: str
    stage_mid_feedback: str
    stage_down_feedback: str
    body_alignment_threshold: float | None = None
    head_drop_threshold: float | None = None
    head_drop_min_main_angle_threshold: float | None = None
    torso_lean_threshold: float | None = None
    knee_forward_ratio_threshold: float | None = None
    heel_lift_ratio_threshold: float | None = None


EXERCISE_CONFIGS = {
    exercise_type: ExerciseConfig(**exercise_config)
    for exercise_type, exercise_config in config.EXERCISE_RUNTIME_CONFIGS.items()
}


class ExerciseSession:
    def __init__(self, exercise_type: str):
        if exercise_type not in EXERCISE_CONFIGS:
            raise ValueError(f"Unsupported exercise type: {exercise_type}")

        self.config = EXERCISE_CONFIGS[exercise_type]
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
        )
        self._position_candidate = None
        self._position_candidate_frames = 0
        self._rolling_values: dict[str, deque[float]] = {}
        self.reset()

    def reset(self, target_reps: int | None = 10):
        self.rep_count = 0
        self.target_reps = None if target_reps in {None, 0} else max(1, int(target_reps))
        self.position = None
        self.stage = "unknown"
        self.last_status = "-"
        self.feedback = self._default_feedback()
        self.history = []
        self.completed = False
        self.ready_position_seen = not config.READY_POSITION_REQUIRED
        self.last_selected_side = None
        self.rep_active = False
        self.bottom_reached = False
        self._position_candidate = None
        self._position_candidate_frames = 0
        self._rolling_values = {
            "main_angle": deque(maxlen=config.ANGLE_SMOOTHING_WINDOW),
            "body_angle": deque(maxlen=config.ANGLE_SMOOTHING_WINDOW),
            "head_drop": deque(maxlen=config.ANGLE_SMOOTHING_WINDOW),
            "torso_lean": deque(maxlen=config.ANGLE_SMOOTHING_WINDOW),
            "knee_forward_ratio": deque(maxlen=config.ANGLE_SMOOTHING_WINDOW),
            "heel_lift_ratio": deque(maxlen=config.ANGLE_SMOOTHING_WINDOW),
            "lateral_ratio": deque(maxlen=config.LATERAL_RATIO_SMOOTHING_WINDOW),
            "torso_ratio": deque(maxlen=config.LATERAL_RATIO_SMOOTHING_WINDOW),
        }
        self._reset_rep_window()

    def process_frame(self, frame, draw_pose: bool = False):
        output_frame = frame.copy() if draw_pose else frame

        if self.completed:
            return self._build_result(output_frame, debug={})

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            self.stage = "no_pose"
            self.feedback = config.MISSING_POSE_FEEDBACK
            return self._build_result(output_frame, debug={})

        landmarks = results.pose_landmarks.landmark
        side = select_body_side(
            landmarks,
            self.config.required_points,
            config.LANDMARK_VISIBILITY_THRESHOLD,
            optional_point_names=self.config.optional_points,
            preferred_side=self.last_selected_side,
            side_switch_margin=config.SIDE_SWITCH_VISIBILITY_MARGIN,
            min_point_visibility=config.LANDMARK_MIN_POINT_VISIBILITY_THRESHOLD,
        )
        if side is None:
            self.stage = "low_visibility"
            self.feedback = config.LOW_VISIBILITY_FEEDBACK
            if draw_pose:
                self._draw_pose(output_frame, results)
            return self._build_result(output_frame, debug={})

        self.last_selected_side = side.side
        torso_ratio = self._smooth("torso_ratio", estimate_torso_size_ratio(landmarks))
        if torso_ratio < config.MIN_TORSO_SIZE_RATIO:
            self.stage = "too_far"
            self.feedback = config.TOO_FAR_FEEDBACK
            if draw_pose:
                self._draw_pose(output_frame, results)
                self._draw_header(output_frame, side.side, 0.0)
            return self._build_result(
                output_frame,
                debug={"selected_side": side.side, "torso_ratio": torso_ratio},
            )

        side_view_ok, lateral_ratio = is_side_view(
            landmarks,
            config.SIDE_VIEW_MAX_LATERAL_RATIO,
        )
        smoothed_lateral_ratio = self._smooth("lateral_ratio", lateral_ratio)
        if config.SIDE_VIEW_REQUIRED and smoothed_lateral_ratio > config.SIDE_VIEW_MAX_LATERAL_RATIO:
            self.stage = "invalid_view"
            self.feedback = config.SIDE_VIEW_FEEDBACK
            if draw_pose:
                self._draw_pose(output_frame, results)
                self._draw_header(output_frame, side.side, smoothed_lateral_ratio)
            return self._build_result(
                output_frame,
                debug={
                    "selected_side": side.side,
                    "lateral_ratio": smoothed_lateral_ratio,
                    "torso_ratio": torso_ratio,
                },
            )

        features = self._extract_features(side)
        smoothed = {name: self._smooth(name, value) for name, value in features.items()}
        self._update_stage(smoothed["main_angle"])
        transition = self._update_position(smoothed["main_angle"])

        if not self.ready_position_seen:
            if transition == "up" or self.position == "up":
                self.ready_position_seen = True
                self.feedback = self.config.stage_up_feedback
            else:
                self.feedback = self.config.ready_pose_feedback
            if draw_pose:
                self._draw_pose(output_frame, results)
                self._draw_metrics(output_frame, side, smoothed, smoothed_lateral_ratio)
            return self._build_result(
                output_frame,
                debug={
                    "selected_side": side.side,
                    "lateral_ratio": smoothed_lateral_ratio,
                    "torso_ratio": torso_ratio,
                    **smoothed,
                },
            )

        self._update_rep_window(smoothed)

        if transition == "down":
            self.bottom_reached = True
        elif transition == "up":
            if (
                self.rep_active
                and self.bottom_reached
                and self._rep_metrics["frame_count"] >= config.MIN_REP_WINDOW_FRAMES
            ):
                self._finalize_rep()
            self._reset_rep_window()

        if draw_pose:
            self._draw_pose(output_frame, results)
            self._draw_metrics(output_frame, side, smoothed, smoothed_lateral_ratio)

        return self._build_result(
            output_frame,
            debug={
                "selected_side": side.side,
                "lateral_ratio": smoothed_lateral_ratio,
                "torso_ratio": torso_ratio,
                **smoothed,
            },
        )

    def get_history(self):
        return self.history

    def _default_feedback(self) -> str:
        return self.config.default_feedback

    def _reset_rep_window(self):
        self.rep_active = False
        self.bottom_reached = False
        self._rep_metrics = {
            "min_main_angle": 180.0,
            "min_body_angle": 180.0,
            "max_head_drop": 0.0,
            "max_torso_lean": 0.0,
            "max_knee_forward_ratio": 0.0,
            "max_heel_lift_ratio": 0.0,
            "frame_count": 0,
            "body_alignment_violations": 0,
            "head_drop_violations": 0,
            "torso_lean_violations": 0,
            "knee_forward_violations": 0,
            "heel_lift_violations": 0,
        }

    def _smooth(self, key: str, value: float) -> float:
        values = self._rolling_values[key]
        values.append(value)
        return sum(values) / len(values)

    def _update_stage(self, main_angle: float):
        if main_angle >= self.config.top_angle:
            self.stage = "up"
            self.feedback = self.config.stage_up_feedback
        elif main_angle <= self.config.bottom_angle:
            self.stage = "down"
            self.feedback = self.config.stage_down_feedback
        else:
            self.stage = "mid"
            self.feedback = self.config.stage_mid_feedback

    def _update_rep_window(self, smoothed: dict[str, float]):
        if not self.rep_active and smoothed["main_angle"] < self.config.track_start_angle:
            self.rep_active = True

        if not self.rep_active:
            return

        self._rep_metrics["frame_count"] += 1

        self._rep_metrics["min_main_angle"] = min(
            self._rep_metrics["min_main_angle"], smoothed["main_angle"]
        )
        self._rep_metrics["min_body_angle"] = min(
            self._rep_metrics["min_body_angle"], smoothed["body_angle"]
        )
        self._rep_metrics["max_head_drop"] = max(
            self._rep_metrics["max_head_drop"], smoothed["head_drop"]
        )
        self._rep_metrics["max_torso_lean"] = max(
            self._rep_metrics["max_torso_lean"], smoothed["torso_lean"]
        )
        self._rep_metrics["max_knee_forward_ratio"] = max(
            self._rep_metrics["max_knee_forward_ratio"], smoothed["knee_forward_ratio"]
        )
        self._rep_metrics["max_heel_lift_ratio"] = max(
            self._rep_metrics["max_heel_lift_ratio"], smoothed["heel_lift_ratio"]
        )
        if (
            self.config.body_alignment_threshold is not None
            and smoothed["body_angle"] < self.config.body_alignment_threshold
        ):
            self._rep_metrics["body_alignment_violations"] += 1
        if (
            self.config.head_drop_threshold is not None
            and smoothed["head_drop"] > self.config.head_drop_threshold
        ):
            self._rep_metrics["head_drop_violations"] += 1
        if (
            self.config.torso_lean_threshold is not None
            and smoothed["torso_lean"] > self.config.torso_lean_threshold
        ):
            self._rep_metrics["torso_lean_violations"] += 1
        if (
            self.config.knee_forward_ratio_threshold is not None
            and smoothed["knee_forward_ratio"] > self.config.knee_forward_ratio_threshold
        ):
            self._rep_metrics["knee_forward_violations"] += 1
        if (
            self.config.heel_lift_ratio_threshold is not None
            and smoothed["heel_lift_ratio"] > self.config.heel_lift_ratio_threshold
        ):
            self._rep_metrics["heel_lift_violations"] += 1

    def _update_position(self, main_angle: float):
        raw_position = None
        if main_angle >= self.config.top_angle:
            raw_position = "up"
        elif main_angle <= self.config.bottom_angle:
            raw_position = "down"

        if raw_position is None:
            self._position_candidate = None
            self._position_candidate_frames = 0
            return None

        if raw_position == self.position:
            self._position_candidate = None
            self._position_candidate_frames = 0
            return None

        if raw_position != self._position_candidate:
            self._position_candidate = raw_position
            self._position_candidate_frames = 1
            return None

        self._position_candidate_frames += 1
        if self._position_candidate_frames < config.STATE_STABLE_FRAMES:
            return None

        self.position = raw_position
        self._position_candidate = None
        self._position_candidate_frames = 0
        return raw_position

    def _extract_features(self, side: SideLandmarks):
        shoulder = (side.points["shoulder"].x, side.points["shoulder"].y)
        hip = (side.points["hip"].x, side.points["hip"].y)
        ankle = (side.points["ankle"].x, side.points["ankle"].y)

        if self.config.exercise_type == "push_up":
            elbow = (side.points["elbow"].x, side.points["elbow"].y)
            wrist = (side.points["wrist"].x, side.points["wrist"].y)
            ear_point = side.points.get("ear")
            head_drop = 0.0
            if ear_point is not None:
                head_drop = safe_ratio(
                    abs(ear_point.y - side.points["shoulder"].y),
                    abs(side.points["shoulder"].y - side.points["hip"].y),
                    default=1.0,
                )
            return {
                "main_angle": calculate_angle(shoulder, elbow, wrist),
                "body_angle": calculate_angle(shoulder, hip, ankle),
                "head_drop": head_drop,
                "torso_lean": 0.0,
                "knee_forward_ratio": 0.0,
                "heel_lift_ratio": 0.0,
            }

        knee = (side.points["knee"].x, side.points["knee"].y)
        heel_point = side.points.get("heel")
        foot_index_point = side.points.get("foot_index")
        heel_lift_ratio = 0.0
        if heel_point is not None and foot_index_point is not None:
            heel_lift_ratio = safe_ratio(
                max(0.0, foot_index_point.y - heel_point.y),
                abs(side.points["hip"].y - side.points["ankle"].y),
                default=0.0,
            )
        return {
            "main_angle": calculate_angle(hip, knee, ankle),
            "body_angle": 180.0,
            "head_drop": 0.0,
            "torso_lean": calculate_vertical_lean(shoulder, hip),
            "knee_forward_ratio": safe_ratio(
                abs(knee[0] - ankle[0]),
                abs(hip[0] - ankle[0]),
                default=0.0,
            ),
            "heel_lift_ratio": heel_lift_ratio,
        }

    def _error_label(self, error_code: str) -> str:
        for item in self.config.error_catalog:
            if item["code"] == error_code:
                return item["label"]
        return error_code

    def _finalize_rep(self):
        passed, message, metrics, error_codes, error_labels = self._evaluate_rep()
        self.rep_count += 1
        self.last_status = "Benar" if passed else "Salah"
        self.feedback = message
        self.history.append(
            {
                "rep": self.rep_count,
                "status": "benar" if passed else "salah",
                "detail": message,
                "error_codes": error_codes,
                "error_labels": error_labels,
                "primary_error_code": error_codes[0] if error_codes else None,
                "metrics": metrics,
            }
        )
        if self.target_reps is not None and self.rep_count >= self.target_reps:
            self.completed = True

    def _evaluate_rep(self):
        frame_count = max(self._rep_metrics["frame_count"], 1)
        violation_threshold = config.FORM_VIOLATION_RATIO_THRESHOLD
        if self.config.exercise_type == "push_up":
            error_codes = []
            if self._rep_metrics["min_main_angle"] > self.config.rom_threshold:
                error_codes.append("tidak_full_rom")
            if (
                self.config.body_alignment_threshold is not None
                and self._rep_metrics["body_alignment_violations"] / frame_count
                >= violation_threshold
            ):
                error_codes.append("badan_bungkuk")
            if (
                self.config.head_drop_threshold is not None
                and self.config.head_drop_min_main_angle_threshold is not None
                and self._rep_metrics["max_head_drop"] >= self.config.head_drop_threshold
                and self._rep_metrics["min_main_angle"]
                >= self.config.head_drop_min_main_angle_threshold
            ):
                error_codes.append("traps_naik")
            metrics = {
                "min_elbow_angle": round(self._rep_metrics["min_main_angle"], 2),
                "min_body_angle": round(self._rep_metrics["min_body_angle"], 2),
                "max_traps_raise_ratio": round(self._rep_metrics["max_head_drop"], 4),
                "form_violation_ratio": round(
                    self._rep_metrics["body_alignment_violations"] / frame_count, 4
                ),
                "traps_raise_violation_ratio": round(
                    self._rep_metrics["head_drop_violations"] / frame_count, 4
                ),
                "frame_count": frame_count,
            }
        else:
            error_codes = []
            if self._rep_metrics["min_main_angle"] > self.config.rom_threshold:
                error_codes.append("tidak_full_rom")
            if (
                self.config.torso_lean_threshold is not None
                and self._rep_metrics["torso_lean_violations"] / frame_count
                >= violation_threshold
            ):
                error_codes.append("badan_bungkuk")
            if (
                self.config.knee_forward_ratio_threshold is not None
                and self._rep_metrics["knee_forward_violations"] / frame_count
                >= violation_threshold
            ):
                error_codes.append("lutut_maju")
            if (
                self.config.heel_lift_ratio_threshold is not None
                and self._rep_metrics["heel_lift_violations"] / frame_count
                >= violation_threshold
            ):
                error_codes.append("kaki_jinjit")
            metrics = {
                "min_knee_angle": round(self._rep_metrics["min_main_angle"], 2),
                "max_torso_lean": round(self._rep_metrics["max_torso_lean"], 2),
                "max_knee_forward_ratio": round(
                    self._rep_metrics["max_knee_forward_ratio"], 4
                ),
                "max_heel_lift_ratio": round(self._rep_metrics["max_heel_lift_ratio"], 4),
                "torso_lean_violation_ratio": round(
                    self._rep_metrics["torso_lean_violations"] / frame_count, 4
                ),
                "knee_forward_violation_ratio": round(
                    self._rep_metrics["knee_forward_violations"] / frame_count, 4
                ),
                "heel_lift_violation_ratio": round(
                    self._rep_metrics["heel_lift_violations"] / frame_count, 4
                ),
                "frame_count": frame_count,
            }

        error_labels = [self._error_label(error_code) for error_code in error_codes]
        if error_labels:
            return False, " | ".join(error_labels), metrics, error_codes, error_labels
        return True, "Teknik benar", metrics, [], []

    def _draw_pose(self, frame, results):
        self.mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(255, 50, 50), thickness=2, circle_radius=3),
            self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2),
        )

    def _draw_header(self, frame, selected_side: str, lateral_ratio: float):
        cv2.putText(
            frame,
            f"Side: {selected_side} | View: {lateral_ratio:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    def _draw_metrics(self, frame, side: SideLandmarks, smoothed: dict[str, float], lateral_ratio: float):
        width = frame.shape[1]
        height = frame.shape[0]
        joint_name = "elbow" if self.config.exercise_type == "push_up" else "knee"
        joint_xy = as_pixel(
            (side.points[joint_name].x, side.points[joint_name].y),
            width,
            height,
        )
        self._draw_header(frame, side.side, lateral_ratio)
        cv2.putText(
            frame,
            f"Main: {int(smoothed['main_angle'])}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            self.last_status,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if self.last_status == "Benar" else (0, 200, 255),
            2,
        )
        cv2.putText(
            frame,
            f"{int(smoothed['main_angle'])} deg",
            (joint_xy[0] + 10, joint_xy[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 230, 0),
            2,
        )

    def _build_result(self, frame, debug: dict):
        return {
            "frame": frame,
            "rep_count": self.rep_count,
            "last_status": self.last_status,
            "stage": self.stage,
            "feedback": self.feedback,
            "completed": self.completed,
            "history": self.history,
            "debug": debug,
        }
