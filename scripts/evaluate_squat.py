import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.video import evaluate_labeled_videos, load_manifest_video_specs


def parse_args():
    parser = argparse.ArgumentParser(description="Run squat evaluation from dataset manifest.")
    parser.add_argument("--manifest", default=config.DATASET_MANIFEST_PATH, help="Path to dataset manifest CSV.")
    parser.add_argument("--split", default=None, help="Optional split filter from manifest.")
    parser.add_argument("--run-name", default="squat_eval", help="Run name used for artifact directory.")
    parser.add_argument("--max-reps", type=int, default=None, help="Optional max reps cap per video.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of videos to evaluate.")
    parser.add_argument(
        "--exclude-issue-flags",
        default="duplicate_hash,basename_conflict",
        help="Comma-separated issue_flags to exclude from benchmark.",
    )
    parser.add_argument("--output-dir", default=config.EVALUATIONS_DIR, help="Evaluation artifact root directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    exclude_issue_flags = {
        value.strip() for value in str(args.exclude_issue_flags).split(",") if value.strip()
    }
    video_specs, validation_issues = load_manifest_video_specs(
        manifest_path=args.manifest,
        exercise_type="squat",
        split=args.split,
        labeled_only=True,
        exclude_issue_flags=exclude_issue_flags,
    )
    if not video_specs:
        raise RuntimeError("Tidak ada video squat berlabel yang lolos filter manifest.")
    if args.limit:
        video_specs = video_specs[: args.limit]

    summary = evaluate_labeled_videos(
        video_specs=video_specs,
        exercise_type="squat",
        title="SQUAT DETECTION",
        max_reps=args.max_reps,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        manifest_validation_issues=validation_issues,
        run_name=args.run_name,
    )
    print(f"Output: {summary['output_dir']}")
    print(f"Videos: {summary['videos_evaluated']}")
    print(f"Videos succeeded: {summary['videos_succeeded']}")
    print(f"Videos failed: {summary['videos_failed']}")
    print(f"Rep predictions: {summary['rep_predictions']}")
    print(f"Accuracy: {summary['rep_quality_metrics']['accuracy']:.4f}")
    print(f"F1: {summary['rep_quality_metrics']['f1']:.4f}")
    print(f"Confusion matrix: {summary['rep_quality_metrics']['confusion_matrix']}")
    print(f"Per-video CSV: {summary['artifact_files']['per_video_results_csv']}")
    print(f"Per-rep CSV: {summary['artifact_files']['per_rep_results_csv']}")
    print(f"Failed videos JSON: {summary['artifact_files']['failed_videos_json']}")
    print(f"Manifest validation issues: {len(validation_issues)}")


if __name__ == "__main__":
    main()
