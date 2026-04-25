import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.video import evaluate_labeled_videos, load_manifest_video_specs


def parse_args():
    parser = argparse.ArgumentParser(description="Run push-up evaluation from dataset manifest.")
    parser.add_argument("--manifest", default=config.DATASET_MANIFEST_PATH, help="Path to dataset manifest CSV.")
    parser.add_argument("--split", default=None, help="Optional split filter from manifest.")
    parser.add_argument("--run-name", default="push_up_eval", help="Run name used for artifact directory.")
    parser.add_argument("--max-reps", type=int, default=None, help="Optional max reps cap per video.")
    parser.add_argument("--output-dir", default=config.EVALUATIONS_DIR, help="Evaluation artifact root directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    video_specs, validation_issues = load_manifest_video_specs(
        manifest_path=args.manifest,
        exercise_type="push_up",
        split=args.split,
        labeled_only=True,
    )
    if not video_specs:
        raise RuntimeError("Tidak ada video push-up berlabel yang lolos filter manifest.")

    summary = evaluate_labeled_videos(
        video_specs=video_specs,
        exercise_type="push_up",
        title="PUSH-UP DETECTION",
        max_reps=args.max_reps,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        manifest_validation_issues=validation_issues,
        run_name=args.run_name,
    )
    print(f"Output: {summary['output_dir']}")
    print(f"Videos: {summary['videos_evaluated']}")
    print(f"Rep predictions: {summary['rep_predictions']}")
    print(f"Manifest validation issues: {len(validation_issues)}")


if __name__ == "__main__":
    main()
