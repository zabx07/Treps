from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional runtime dependency
    plt = None


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "evaluation"


def evaluate_model(y_true, y_pred, title="MODEL", output_dir="artifacts/evaluations", show=False):
    labels = ["benar", "salah"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    label_support = {label: int(sum(1 for value in y_true if value == label)) for label in labels}
    metrics = {
        "title": title,
        "samples": len(y_true),
        "labels": labels,
        "label_support": label_support,
        "confusion_matrix": cm.tolist(),
        "accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "precision": precision_score(y_true, y_pred, pos_label="benar", zero_division=0) if y_true else 0.0,
        "recall": recall_score(y_true, y_pred, pos_label="benar", zero_division=0) if y_true else 0.0,
        "f1": f1_score(y_true, y_pred, pos_label="benar", zero_division=0) if y_true else 0.0,
        "y_true": list(y_true),
        "y_pred": list(y_pred),
        "plot_generated": plt is not None,
        "plot_status": "available" if plt is not None else "skipped_missing_matplotlib",
    }

    print(f"\n===== EVALUASI {title} =====")
    print(f"Samples   : {metrics['samples']}")
    print(f"Support   : {metrics['label_support']}")
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nAccuracy  : {metrics['accuracy']:.2f}")
    print(f"Precision : {metrics['precision']:.2f}")
    print(f"Recall    : {metrics['recall']:.2f}")
    print(f"F1 Score  : {metrics['f1']:.2f}")

    slug = slugify(title)
    png_path = output_dir / f"{slug}_confusion_matrix.png"
    json_path = output_dir / f"{slug}_metrics.json"

    if plt is not None:
        plt.figure(figsize=(5, 4))
        plt.imshow(cm)
        plt.title(f"Confusion Matrix - {title}")
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(png_path)
        if show:
            plt.show(block=True)
        plt.close()
        print(f"Confusion matrix tersimpan di {png_path}")
    else:
        print("Confusion matrix plot dilewati karena matplotlib tidak tersedia.")

    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics tersimpan di {json_path}")
    return metrics
