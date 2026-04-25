import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.evaluator import evaluate_model
from core.video import load_manifest_video_specs


def check_manifest():
    manifest_path = PROJECT_ROOT / config.DATASET_MANIFEST_PATH
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest tidak ditemukan: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
    missing_fields = [field for field in config.MANIFEST_REQUIRED_FIELDS if field not in headers]
    if missing_fields:
        raise ValueError(f"Manifest field wajib belum lengkap: {', '.join(missing_fields)}")

    push_up_specs, push_up_issues = load_manifest_video_specs(
        manifest_path=manifest_path,
        exercise_type="push_up",
        labeled_only=False,
    )
    squat_specs, squat_issues = load_manifest_video_specs(
        manifest_path=manifest_path,
        exercise_type="squat",
        labeled_only=False,
    )
    return {
        "manifest_path": str(manifest_path),
        "push_up_rows": len(push_up_specs),
        "squat_rows": len(squat_specs),
        "validation_issues": len(push_up_issues) + len(squat_issues),
    }


def check_issue_summary():
    issues_path = PROJECT_ROOT / config.DATASET_ISSUES_PATH
    if not issues_path.exists():
        raise FileNotFoundError(f"Issue summary tidak ditemukan: {issues_path}")

    summary = json.loads(issues_path.read_text(encoding="utf-8"))
    required_keys = {
        "total_videos",
        "counts_by_exercise",
        "counts_by_label",
        "metadata_missing_counts",
        "issue_flag_counts",
        "validation",
        "ground_truth_gaps",
    }
    missing_keys = sorted(required_keys.difference(summary))
    if missing_keys:
        raise ValueError(f"Issue summary belum lengkap: {', '.join(missing_keys)}")
    return {
        "issues_path": str(issues_path),
        "total_videos": summary["total_videos"],
        "issue_flags": summary["issue_flag_counts"],
    }


def run_metric_smoke():
    output_dir = PROJECT_ROOT / config.EVALUATIONS_DIR / "smoke_test"
    metrics = evaluate_model(
        y_true=["benar", "benar", "salah", "salah"],
        y_pred=["benar", "salah", "salah", "salah"],
        title="SMOKE TEST",
        output_dir=output_dir,
        show=False,
    )
    metrics_path = output_dir / "smoke_test_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics smoke artifact tidak dibuat: {metrics_path}")
    return {
        "metrics_path": str(metrics_path),
        "accuracy": metrics["accuracy"],
        "plot_status": metrics["plot_status"],
    }


def main():
    manifest_result = check_manifest()
    issues_result = check_issue_summary()
    metrics_result = run_metric_smoke()

    print("Manifest OK")
    print(manifest_result)
    print("Issue summary OK")
    print(issues_result)
    print("Evaluator smoke OK")
    print(metrics_result)


if __name__ == "__main__":
    main()
