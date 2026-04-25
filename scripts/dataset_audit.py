from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

DATASET_ROOT = PROJECT_ROOT / "dataset"
ARTIFACTS_DIR = PROJECT_ROOT / config.ARTIFACTS_DIR
MANIFEST_PATH = PROJECT_ROOT / config.DATASET_MANIFEST_PATH
ISSUES_PATH = PROJECT_ROOT / config.DATASET_ISSUES_PATH

EXERCISE_NAME_MAP = {
    "Pushup": "push_up",
    "Squat": "squat",
}
NON_EMPTY_MANIFEST_FIELDS = {
    "file_path",
    "file_name",
    "exercise_type",
    "label_main",
    "split_group",
    "label_group",
    "size_bytes",
    "sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_exercise_name(raw_name: str) -> str:
    return EXERCISE_NAME_MAP.get(raw_name, raw_name.lower())


def classify_path(relative_path: Path) -> dict[str, object]:
    parts = relative_path.parts
    raw_exercise = parts[0] if len(parts) > 0 else ""
    exercise_type = normalize_exercise_name(raw_exercise)
    label_group = ""
    label_main = "unlabeled"
    error_tags: list[str] = []

    if len(parts) >= 2 and parts[1] == "gerakan_benar":
        label_group = parts[1]
        label_main = "benar"
    elif len(parts) >= 2 and parts[1] == "gerakan_salah":
        label_group = parts[1]
        label_main = "salah"
        if len(parts) >= 3:
            error_tags.append(parts[2])
    elif len(parts) >= 2:
        label_group = "unlabeled_root"

    return {
        "exercise_type": exercise_type,
        "exercise_folder": raw_exercise,
        "label_group": label_group,
        "label_main": label_main,
        "error_tags": error_tags,
    }


def infer_clip_source(stem: str) -> str:
    part_match = re.match(r"(.+?)_part_\d+$", stem)
    if part_match:
        return part_match.group(1)
    return ""


def build_root_source_index() -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = defaultdict(dict)
    for path in sorted(DATASET_ROOT.rglob("*.mp4")):
        relative_path = path.relative_to(PROJECT_ROOT)
        labels = classify_path(relative_path.relative_to("dataset"))
        if labels["label_main"] != "unlabeled":
            continue
        index[labels["exercise_type"]][path.stem] = str(relative_path).replace("\\", "/")
    return index


def load_existing_manifest_annotations() -> dict[str, dict[str, str]]:
    manifest_annotations: dict[str, dict[str, str]] = {}
    if not MANIFEST_PATH.exists():
        return manifest_annotations

    with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "file_path" not in reader.fieldnames:
            return manifest_annotations

        for row in reader:
            file_path = row.get("file_path", "")
            if not file_path:
                continue
            manifest_annotations[file_path] = {
                field_name: row.get(field_name, "")
                for field_name in config.MANIFEST_MANUAL_FIELDS
            }
    return manifest_annotations


def build_manifest_rows() -> list[dict]:
    root_source_index = build_root_source_index()
    existing_annotations = load_existing_manifest_annotations()
    rows = []

    for path in sorted(DATASET_ROOT.rglob("*.mp4")):
        relative_path = path.relative_to(PROJECT_ROOT)
        labels = classify_path(relative_path.relative_to("dataset"))
        clip_source = infer_clip_source(path.stem)
        source_video = ""
        if clip_source and clip_source in root_source_index[labels["exercise_type"]]:
            source_video = root_source_index[labels["exercise_type"]][clip_source]
        elif labels["label_main"] == "unlabeled":
            clip_source = path.stem
            source_video = str(relative_path).replace("\\", "/")

        row = {
            "file_path": str(relative_path).replace("\\", "/"),
            "file_name": path.name,
            "exercise_type": labels["exercise_type"],
            "label_main": labels["label_main"],
            "error_tags": ";".join(labels["error_tags"]),
            "expected_reps": "",
            "subject_id": "",
            "camera_view": "",
            "occlusion_level": "",
            "clip_source": clip_source,
            "source_video": source_video,
            "split": "",
            "notes": "",
            "split_group": clip_source or path.stem,
            "label_group": labels["label_group"],
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "duplicate_hash_group": "",
            "basename_conflict_group": "",
            "issue_flags": "",
        }
        existing_row = existing_annotations.get(row["file_path"], {})
        for field_name in config.MANIFEST_MANUAL_FIELDS:
            existing_value = existing_row.get(field_name, "")
            if existing_value not in {None, ""}:
                row[field_name] = existing_value
        rows.append(row)

    return rows


def assign_issue_flags(rows: list[dict]) -> tuple[dict, dict, dict]:
    duplicate_hash_groups = defaultdict(list)
    basename_groups = defaultdict(list)
    split_group_label_sets = defaultdict(set)

    for row in rows:
        duplicate_hash_groups[row["sha256"]].append(row["file_path"])
        basename_groups[row["file_name"]].append(row["file_path"])
        split_group_label_sets[(row["exercise_type"], row["split_group"])].add(
            (row["label_main"], row["error_tags"])
        )

    duplicate_hashes = {
        file_hash: sorted(paths)
        for file_hash, paths in duplicate_hash_groups.items()
        if len(paths) > 1
    }
    basename_conflicts = {
        file_name: sorted(paths)
        for file_name, paths in basename_groups.items()
        if len(paths) > 1
    }
    split_group_conflicts = {
        f"{exercise_type}:{split_group}": sorted(list(label_pairs))
        for (exercise_type, split_group), label_pairs in split_group_label_sets.items()
        if split_group and len(label_pairs) > 1
    }

    duplicate_group_ids = {file_hash: f"dup_hash_{index:03d}" for index, file_hash in enumerate(sorted(duplicate_hashes), start=1)}
    basename_group_ids = {
        file_name: f"basename_conflict_{index:03d}"
        for index, file_name in enumerate(sorted(basename_conflicts), start=1)
    }

    for row in rows:
        issue_flags = []
        if row["label_main"] == "unlabeled":
            issue_flags.append("unlabeled")
        if row["sha256"] in duplicate_hashes:
            row["duplicate_hash_group"] = duplicate_group_ids[row["sha256"]]
            issue_flags.append("duplicate_hash")
        if row["file_name"] in basename_conflicts:
            row["basename_conflict_group"] = basename_group_ids[row["file_name"]]
            issue_flags.append("basename_conflict")
        split_group_key = f"{row['exercise_type']}:{row['split_group']}"
        if split_group_key in split_group_conflicts:
            issue_flags.append("split_group_multi_label")
        row["issue_flags"] = ";".join(issue_flags)

    return duplicate_hashes, basename_conflicts, split_group_conflicts


def validate_manifest_rows(rows: list[dict]) -> dict:
    validation = {
        "missing_files": [],
        "missing_required_fields": defaultdict(list),
        "invalid_expected_reps": [],
        "invalid_label_main": [],
        "invalid_exercise_type": [],
    }

    for row in rows:
        file_path = PROJECT_ROOT / row["file_path"]
        if not file_path.exists():
            validation["missing_files"].append(row["file_path"])

        for field_name in config.MANIFEST_REQUIRED_FIELDS:
            if field_name not in row:
                validation["missing_required_fields"][field_name].append(row["file_path"])
                continue
            if field_name not in NON_EMPTY_MANIFEST_FIELDS:
                continue
            if row[field_name] == "":
                validation["missing_required_fields"][field_name].append(row["file_path"])

        if row["expected_reps"] not in {"", None}:
            try:
                expected_reps = int(row["expected_reps"])
                if expected_reps < 0:
                    raise ValueError("negative")
            except (TypeError, ValueError):
                validation["invalid_expected_reps"].append(row["file_path"])

        if row["label_main"] not in {"benar", "salah", "unlabeled"}:
            validation["invalid_label_main"].append(row["file_path"])

        if row["exercise_type"] not in config.EXERCISE_RUNTIME_CONFIGS:
            validation["invalid_exercise_type"].append(row["file_path"])

    validation["missing_required_fields"] = dict(validation["missing_required_fields"])
    return validation


def build_issue_summary(rows: list[dict], duplicate_hashes: dict, basename_conflicts: dict, split_group_conflicts: dict, validation: dict) -> dict:
    counts_by_exercise = Counter(row["exercise_type"] for row in rows)
    counts_by_label = Counter(f"{row['exercise_type']}:{row['label_main']}" for row in rows)
    counts_by_error_tag = Counter(
        f"{row['exercise_type']}:{error_tag}"
        for row in rows
        for error_tag in ([tag for tag in row["error_tags"].split(";") if tag] or [])
    )
    metadata_missing_counts = {
        field_name: sum(1 for row in rows if row.get(field_name, "") == "")
        for field_name in [*config.MANIFEST_MANUAL_FIELDS, "clip_source", "source_video"]
    }
    unlabeled_root_files = [row["file_path"] for row in rows if row["label_main"] == "unlabeled"]

    issue_flags_counter = Counter()
    for row in rows:
        for flag in [flag for flag in row["issue_flags"].split(";") if flag]:
            issue_flags_counter[flag] += 1

    return {
        "total_videos": len(rows),
        "counts_by_exercise": dict(sorted(counts_by_exercise.items())),
        "counts_by_label": dict(sorted(counts_by_label.items())),
        "counts_by_error_tag": dict(sorted(counts_by_error_tag.items())),
        "metadata_missing_counts": metadata_missing_counts,
        "issue_flag_counts": dict(sorted(issue_flags_counter.items())),
        "unlabeled_root_files": unlabeled_root_files,
        "duplicate_hashes": duplicate_hashes,
        "duplicate_or_conflicting_basenames": basename_conflicts,
        "split_group_multi_label_conflicts": split_group_conflicts,
        "validation": validation,
        "ground_truth_gaps": {
            "expected_reps": "UNKNOWN: field tersedia di manifest, tetapi belum terisi dari dataset saat ini.",
            "subject_id": "UNKNOWN: tidak bisa diinfer aman dari nama file/folder saat ini.",
            "camera_view": "UNKNOWN: tidak ada metadata view eksplisit di dataset saat ini.",
            "occlusion_level": "UNKNOWN: tidak ada anotasi occlusion eksplisit di dataset saat ini.",
            "split": "UNKNOWN: belum ada split evaluasi/train-val-test yang sehat dan eksplisit.",
            "phase_labels": "UNKNOWN: belum ada ground truth fase per frame/per event.",
        },
        "custom_model_readiness": {
            "ready_for_rule_evaluation": True,
            "ready_for_rep_count_benchmark": False,
            "ready_for_phase_model_training": False,
            "blocking_factors": [
                "expected_reps belum terisi",
                "subject_id belum ada",
                "split belum ada",
                "phase/event annotation belum ada",
            ],
        },
    }


def write_manifest(rows: list[dict]):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=config.MANIFEST_REQUIRED_FIELDS + ["duplicate_hash_group", "basename_conflict_group", "issue_flags"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_manifest_rows()
    duplicate_hashes, basename_conflicts, split_group_conflicts = assign_issue_flags(rows)
    validation = validate_manifest_rows(rows)
    issue_summary = build_issue_summary(
        rows,
        duplicate_hashes,
        basename_conflicts,
        split_group_conflicts,
        validation,
    )

    write_manifest(rows)
    ISSUES_PATH.write_text(json.dumps(issue_summary, indent=2), encoding="utf-8")

    print(f"Manifest written to {MANIFEST_PATH}")
    print(f"Issues written to {ISSUES_PATH}")
    print(f"Videos indexed: {len(rows)}")


if __name__ == "__main__":
    main()
