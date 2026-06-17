from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import config

from .evaluator import evaluate_model, slugify


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _mean(values):
    return round(sum(values) / len(values), 4) if values else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _safe_int(value):
    if value in {"", None}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_count_accuracy(expected_reps: int | None, detected_reps: int) -> float | None:
    if expected_reps is None:
        return None
    if expected_reps == 0:
        return 1.0 if detected_reps == 0 else 0.0
    absolute_error = abs(detected_reps - expected_reps)
    return round(max(0.0, 1.0 - (absolute_error / expected_reps)), 4)


def _normalize_error_tags(value: str | None) -> list[str]:
    if not value:
        return []
    return [tag for tag in value.split(";") if tag]


def _validation_issue(row_index: int, file_path: str, field_name: str, message: str) -> dict:
    return {
        "row_index": row_index,
        "file_path": file_path,
        "field": field_name,
        "message": message,
    }


def load_manifest_video_specs(
    manifest_path: str | Path = config.DATASET_MANIFEST_PATH,
    exercise_type: str | None = None,
    split: str | None = None,
    labeled_only: bool = True,
    exclude_issue_flags: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest tidak ditemukan: {manifest_path}. Jalankan scripts/dataset_audit.py terlebih dahulu."
        )

    video_specs = []
    validation_issues = []
    exclude_issue_flags = {flag for flag in (exclude_issue_flags or set()) if flag}

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_fields = [
            field_name for field_name in config.MANIFEST_REQUIRED_FIELDS if field_name not in reader.fieldnames
        ]
        if missing_fields:
            raise ValueError(
                f"Manifest tidak memiliki field wajib: {', '.join(missing_fields)}"
            )

        for row_index, row in enumerate(reader, start=2):
            file_path = row["file_path"]
            row_exercise_type = row["exercise_type"]
            row_label_main = row["label_main"]

            if row_exercise_type not in config.EXERCISE_RUNTIME_CONFIGS:
                validation_issues.append(
                    _validation_issue(
                        row_index,
                        file_path,
                        "exercise_type",
                        f"Nilai exercise_type tidak valid: {row_exercise_type}",
                    )
                )
                continue

            if row_label_main not in {"benar", "salah", "unlabeled"}:
                validation_issues.append(
                    _validation_issue(
                        row_index,
                        file_path,
                        "label_main",
                        f"Nilai label_main tidak valid: {row_label_main}",
                    )
                )
                continue

            expected_reps = _safe_int(row["expected_reps"])
            if row["expected_reps"] not in {"", None} and expected_reps is None:
                validation_issues.append(
                    _validation_issue(
                        row_index,
                        file_path,
                        "expected_reps",
                        f"expected_reps bukan integer valid: {row['expected_reps']}",
                    )
                )

            resolved_path = PROJECT_ROOT / file_path
            if not resolved_path.exists():
                validation_issues.append(
                    _validation_issue(
                        row_index,
                        file_path,
                        "file_path",
                        "Path file tidak ditemukan.",
                    )
                )
                continue

            if exercise_type and row_exercise_type != exercise_type:
                continue

            if split:
                row_split = row["split"].strip()
                if not row_split:
                    continue
                if row_split != split:
                    continue

            if labeled_only and row_label_main not in {"benar", "salah"}:
                continue

            metadata = dict(row)
            metadata["expected_reps"] = expected_reps
            metadata["error_tags"] = _normalize_error_tags(row["error_tags"])
            metadata["issue_flags_list"] = _normalize_error_tags(row.get("issue_flags"))
            if exclude_issue_flags and set(metadata["issue_flags_list"]) & exclude_issue_flags:
                continue
            video_specs.append(
                {
                    "path": resolved_path,
                    "label": row_label_main,
                    "expected_reps": expected_reps,
                    "error_tags": metadata["error_tags"],
                    "metadata": metadata,
                }
            )

    return video_specs, validation_issues


def _status_counts(history: list[dict]) -> dict:
    counts = Counter(rep["status"] for rep in history)
    return {"benar": counts.get("benar", 0), "salah": counts.get("salah", 0)}


def _build_per_video_record(spec: dict, result: dict) -> dict:
    metadata = spec["metadata"]
    status_counts = _status_counts(result["history"])
    expected_reps = spec["expected_reps"]
    detected_reps = result["rep_count"]
    count_metrics = {
        "expected_reps": expected_reps,
        "detected_reps": detected_reps,
        "rep_count_error": None,
        "absolute_rep_count_error": None,
        "false_positive_reps": None,
        "false_negative_reps": None,
        "count_accuracy": None,
    }
    if expected_reps is not None:
        count_metrics = {
            "expected_reps": expected_reps,
            "detected_reps": detected_reps,
            "rep_count_error": detected_reps - expected_reps,
            "absolute_rep_count_error": abs(detected_reps - expected_reps),
            "false_positive_reps": max(0, detected_reps - expected_reps),
            "false_negative_reps": max(0, expected_reps - detected_reps),
            "count_accuracy": _safe_count_accuracy(expected_reps, detected_reps),
        }

    return {
        "file_path": metadata["file_path"],
        "file_name": metadata["file_name"],
        "exercise_type": metadata["exercise_type"],
        "label_main": metadata["label_main"],
        "error_tags": ";".join(metadata["error_tags"]),
        "expected_reps": count_metrics["expected_reps"],
        "detected_reps": count_metrics["detected_reps"],
        "rep_count_error": count_metrics["rep_count_error"],
        "absolute_rep_count_error": count_metrics["absolute_rep_count_error"],
        "false_positive_reps": count_metrics["false_positive_reps"],
        "false_negative_reps": count_metrics["false_negative_reps"],
        "count_accuracy": count_metrics["count_accuracy"],
        "predicted_benar_reps": status_counts["benar"],
        "predicted_salah_reps": status_counts["salah"],
        "no_pose_rate": result["no_pose_rate"],
        "low_visibility_rate": result["low_visibility_rate"],
        "invalid_view_rate": result["invalid_view_rate"],
        "too_far_rate": result.get("too_far_rate", 0.0),
        "fps": result["fps"],
        "processing_seconds": result["processing_seconds"],
        "subject_id": metadata["subject_id"],
        "camera_view": metadata["camera_view"],
        "occlusion_level": metadata["occlusion_level"],
        "clip_source": metadata["clip_source"],
        "source_video": metadata["source_video"],
        "split": metadata["split"],
        "split_group": metadata["split_group"],
        "notes": metadata["notes"],
        "issue_flags": metadata.get("issue_flags", ""),
        "history_length": len(result["history"]),
    }


def _build_per_rep_records(spec: dict, result: dict) -> list[dict]:
    metadata = spec["metadata"]
    records = []
    for rep in result["history"]:
        error_codes = rep.get("error_codes") or []
        error_labels = rep.get("error_labels") or []
        records.append(
            {
                "file_path": metadata["file_path"],
                "file_name": metadata["file_name"],
                "exercise_type": metadata["exercise_type"],
                "expected_label": spec["label"],
                "expected_error_tags": ";".join(spec["error_tags"]),
                "rep_index": rep.get("rep"),
                "predicted_label": rep.get("status"),
                "predicted_detail": rep.get("detail"),
                "predicted_primary_error_code": rep.get("primary_error_code") or "",
                "predicted_error_codes": ";".join(error_codes),
                "predicted_error_labels": ";".join(error_labels),
                "metrics_json": json.dumps(rep.get("metrics", {}), ensure_ascii=False, sort_keys=True),
            }
        )
    return records


def _serialize_video_result(result: dict) -> dict:
    last_result = result.get("last_result") or {}
    serializable_last_result = {
        key: value for key, value in last_result.items() if key != "frame"
    }
    return {
        key: value
        for key, value in {
            **result,
            "last_result": serializable_last_result,
        }.items()
        if key != "frame"
    }


def _summarize_count_metrics(per_video_records: list[dict]) -> dict:
    rows = [row for row in per_video_records if row["expected_reps"] is not None]
    if not rows:
        return {
            "status": "UNKNOWN",
            "reason": "Ground truth expected_reps belum tersedia pada manifest.",
            "videos_with_expected_reps": 0,
        }

    total_expected = sum(row["expected_reps"] for row in rows)
    total_detected = sum(row["detected_reps"] for row in rows)
    total_abs_error = sum(row["absolute_rep_count_error"] for row in rows)
    total_false_positive = sum(row["false_positive_reps"] for row in rows)
    total_false_negative = sum(row["false_negative_reps"] for row in rows)
    return {
        "status": "available",
        "videos_with_expected_reps": len(rows),
        "total_expected_reps": total_expected,
        "total_detected_reps": total_detected,
        "rep_count_error": total_detected - total_expected,
        "mean_rep_count_error": _mean([row["rep_count_error"] for row in rows]),
        "mean_absolute_rep_count_error": _mean(
            [row["absolute_rep_count_error"] for row in rows]
        ),
        "false_positive_reps": total_false_positive,
        "false_negative_reps": total_false_negative,
        "count_accuracy": round(max(0.0, 1.0 - (total_abs_error / total_expected)), 4)
        if total_expected
        else None,
    }


def _summarize_runtime_metrics(per_video_records: list[dict]) -> dict:
    return {
        "mean_fps": _mean([row["fps"] for row in per_video_records]),
        "mean_processing_seconds": _mean(
            [row["processing_seconds"] for row in per_video_records]
        ),
        "mean_no_pose_rate": _mean([row["no_pose_rate"] for row in per_video_records]),
        "mean_low_visibility_rate": _mean(
            [row["low_visibility_rate"] for row in per_video_records]
        ),
        "mean_invalid_view_rate": _mean(
            [row["invalid_view_rate"] for row in per_video_records]
        ),
        "mean_too_far_rate": _mean([row["too_far_rate"] for row in per_video_records]),
        "total_detected_reps": sum(row["detected_reps"] for row in per_video_records),
        "videos_without_reps": sum(1 for row in per_video_records if row["detected_reps"] == 0),
    }


def _aggregate_rows(rows: list[dict]) -> dict:
    return {
        "videos": len(rows),
        "detected_reps": sum(row["detected_reps"] for row in rows),
        "predicted_benar_reps": sum(row["predicted_benar_reps"] for row in rows),
        "predicted_salah_reps": sum(row["predicted_salah_reps"] for row in rows),
        "mean_fps": _mean([row["fps"] for row in rows]),
        "mean_no_pose_rate": _mean([row["no_pose_rate"] for row in rows]),
        "mean_low_visibility_rate": _mean([row["low_visibility_rate"] for row in rows]),
        "mean_invalid_view_rate": _mean([row["invalid_view_rate"] for row in rows]),
        "mean_too_far_rate": _mean([row["too_far_rate"] for row in rows]),
        "count_metrics": _summarize_count_metrics(rows),
    }


def _summarize_predicted_rep_errors(per_rep_records: list[dict]) -> dict:
    if not per_rep_records:
        return {"none": {"reps": 0}}
    groups = defaultdict(list)
    for row in per_rep_records:
        codes = [code for code in row["predicted_error_codes"].split(";") if code]
        if not codes:
            groups["none"].append(row)
            continue
        for code in codes:
            groups[code].append(row)
    return {
        group_name: {
            "reps": len(rows),
            "files": len({row["file_path"] for row in rows}),
        }
        for group_name, rows in sorted(groups.items())
    }


def _summarize_error_tag_metrics(per_rep_records: list[dict]) -> dict:
    labels = sorted(
        {
            tag
            for row in per_rep_records
            for tag in (
                [value for value in row["expected_error_tags"].split(";") if value]
                + [value for value in row["predicted_error_codes"].split(";") if value]
            )
        }
    )
    if not labels:
        return {
            "status": "UNKNOWN",
            "reason": "Ground truth error tag per rep belum tersedia atau tidak ada rep terdeteksi.",
        }

    metrics = {}
    for label in labels:
        tp = fp = fn = tn = 0
        for row in per_rep_records:
            expected_tags = {value for value in row["expected_error_tags"].split(";") if value}
            predicted_tags = {value for value in row["predicted_error_codes"].split(";") if value}
            y_true = label in expected_tags
            y_pred = label in predicted_tags
            if y_true and y_pred:
                tp += 1
            elif y_true:
                fn += 1
            elif y_pred:
                fp += 1
            else:
                tn += 1
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        metrics[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support": tp + fn,
            "predicted_positive": tp + fp,
            "precision": precision,
            "recall": recall,
            "f1": round((2 * precision * recall) / (precision + recall), 4)
            if (precision + recall)
            else 0.0,
        }

    return {
        "status": "limited_clip_level_ground_truth",
        "reason": (
            "Error tag ground truth masih diwariskan dari label level video ke seluruh rep, "
            "jadi metrik ini indikatif dan belum setara anotasi per-rep."
        ),
        "labels": metrics,
    }


def _summarize_by_field(per_video_records: list[dict], field_name: str, empty_label: str = "none") -> dict:
    groups = defaultdict(list)
    for row in per_video_records:
        raw_value = row[field_name]
        if not raw_value:
            groups[empty_label].append(row)
            continue
        values = [value for value in str(raw_value).split(";") if value]
        if not values:
            groups[empty_label].append(row)
            continue
        for value in values:
            groups[value].append(row)
    return {group_name: _aggregate_rows(rows) for group_name, rows in sorted(groups.items())}


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_output_dir(output_root: str | Path, exercise_type: str, title: str, run_name: str | None = None) -> Path:
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_slug = slugify(run_name or title)
    return output_root / exercise_type / f"{timestamp}_{run_slug}"


def run_video_session(
    video_path,
    exercise_type: str,
    max_reps: int | None = None,
    draw_pose: bool = False,
):
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV (cv2) belum terpasang. Jalankan pip install -r requirements.txt.") from exc

    try:
        from .engine import ExerciseSession
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Exercise engine tidak bisa diimpor. Pastikan mediapipe dan dependency runtime terpasang."
        ) from exc

    session = ExerciseSession(exercise_type)
    session.reset(target_reps=max_reps)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video tidak bisa dibuka: {video_path}")

    frame_count = 0
    pose_missing_frames = 0
    low_visibility_frames = 0
    invalid_view_frames = 0
    too_far_frames = 0
    started_at = time.perf_counter()
    last_result = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            last_result = session.process_frame(frame, draw_pose=draw_pose)
            if last_result["stage"] == "no_pose":
                pose_missing_frames += 1
            elif last_result["stage"] == "low_visibility":
                low_visibility_frames += 1
            elif last_result["stage"] == "invalid_view":
                invalid_view_frames += 1
            elif last_result["stage"] == "too_far":
                too_far_frames += 1

            if max_reps is not None and session.rep_count >= max_reps:
                break

        duration = time.perf_counter() - started_at
        detected_reps = len(session.history)

        return {
            "video_path": str(video_path),
            "exercise_type": exercise_type,
            "frame_count": frame_count,
            "processing_seconds": round(duration, 4),
            "fps": round(frame_count / duration, 2) if duration > 0 else 0.0,
            "no_pose_rate": round(pose_missing_frames / frame_count, 4) if frame_count else 0.0,
            "low_visibility_rate": round(low_visibility_frames / frame_count, 4)
            if frame_count
            else 0.0,
            "invalid_view_rate": round(invalid_view_frames / frame_count, 4)
            if frame_count
            else 0.0,
            "too_far_rate": round(too_far_frames / frame_count, 4) if frame_count else 0.0,
            "detected_reps": detected_reps,
            "rep_count": session.rep_count,
            "history": session.history,
            "last_result": last_result,
        }
    finally:
        cap.release()
        try:
            session.pose.close()
        except Exception:
            pass


def evaluate_labeled_videos(
    video_specs: list[dict],
    exercise_type: str,
    title: str,
    max_reps: int | None = None,
    output_dir: str | Path = config.EVALUATIONS_DIR,
    manifest_path: str | Path | None = None,
    manifest_validation_issues: list[dict] | None = None,
    run_name: str | None = None,
):
    output_dir = _build_output_dir(output_dir, exercise_type, title, run_name=run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = []
    y_pred = []
    video_results = []
    per_video_records = []
    per_rep_records = []
    failed_videos = []

    for spec in video_specs:
        try:
            result = run_video_session(
                spec["path"],
                exercise_type=exercise_type,
                max_reps=spec.get("max_reps", max_reps),
                draw_pose=False,
            )
            result["expected_label"] = spec["label"]
            result["expected_reps"] = spec["expected_reps"]
            result["error_tags"] = spec["error_tags"]
            result["metadata"] = spec["metadata"]
            video_results.append(result)
            per_video_records.append(_build_per_video_record(spec, result))
            per_rep_records.extend(_build_per_rep_records(spec, result))

            for rep in result["history"]:
                y_true.append(spec["label"])
                y_pred.append(rep["status"])
        except Exception as exc:
            failed_videos.append(
                {
                    "file_path": str(spec["path"]),
                    "exercise_type": exercise_type,
                    "label": spec["label"],
                    "error_tags": spec["error_tags"],
                    "error": str(exc),
                }
            )

    rep_quality_metrics = evaluate_model(
        y_true,
        y_pred,
        title=f"{title} REP QUALITY",
        output_dir=output_dir,
        show=False,
    )

    _write_csv(output_dir / "per_video_results.csv", per_video_records)
    _write_csv(output_dir / "per_rep_results.csv", per_rep_records)
    _write_csv(
        output_dir / "evaluated_manifest_snapshot.csv",
        [spec["metadata"] for spec in video_specs],
    )

    summary = {
        "title": title,
        "exercise_type": exercise_type,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "dataset_manifest_path": str(manifest_path) if manifest_path else None,
        "videos_evaluated": len(video_specs),
        "videos_succeeded": len(video_results),
        "videos_failed": len(failed_videos),
        "rep_predictions": len(y_pred),
        "rep_quality_metrics": rep_quality_metrics,
        "predicted_rep_error_summary": _summarize_predicted_rep_errors(per_rep_records),
        "error_tag_metrics": _summarize_error_tag_metrics(per_rep_records),
        "runtime_metrics": _summarize_runtime_metrics(per_video_records),
        "count_metrics": _summarize_count_metrics(per_video_records),
        "summary_by_exercise": _summarize_by_field(per_video_records, "exercise_type"),
        "summary_by_error_type": _summarize_by_field(per_video_records, "error_tags"),
        "summary_by_label_main": _summarize_by_field(per_video_records, "label_main"),
        "artifact_files": {
            "per_video_results_csv": str(output_dir / "per_video_results.csv"),
            "per_rep_results_csv": str(output_dir / "per_rep_results.csv"),
            "failed_videos_json": str(output_dir / "failed_videos.json"),
            "summary_json": str(output_dir / "summary.json"),
        },
        "manifest_validation_issues": manifest_validation_issues or [],
        "failed_videos": failed_videos,
        "unknown_metrics": {
            "phase_metrics": "UNKNOWN: dataset belum punya ground truth fase/event per frame.",
            "camera_view_robustness": "UNKNOWN: field camera_view di manifest belum terisi.",
            "occlusion_robustness": "UNKNOWN: field occlusion_level di manifest belum terisi.",
            "subject_generalization": "UNKNOWN: subject_id belum tersedia.",
        },
        "videos": [_serialize_video_result(video_result) for video_result in video_results],
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    failed_path = output_dir / "failed_videos.json"
    failed_path.write_text(json.dumps(failed_videos, indent=2), encoding="utf-8")
    validation_path = output_dir / "manifest_validation_issues.json"
    validation_path.write_text(
        json.dumps(manifest_validation_issues or [], indent=2),
        encoding="utf-8",
    )
    return summary
