# TREPS Fitness Detection

Project ini adalah pipeline evaluasi gerakan berbasis `MediaPipe Pose` untuk `push_up` dan `squat`.
Saat ini sistem memakai landmark MediaPipe + rule engine/FSM bersama di atas `core/`, bukan custom trained model.

## Struktur aktif

```text
fitness-detection2/
  app/         # Flask routes + templates
  core/        # shared pose pipeline, FSM, evaluator
  scripts/     # evaluation, debug, audit, smoke checks
  dataset/     # raw/labeled video dataset
  artifacts/   # output evaluasi dan audit dataset
  run.py       # web runtime entrypoint
  config.py    # central thresholds/config
```

## Jalur aktif

- Web runtime: `run.py` -> `app/__init__.py` -> `app/routes.py` -> `core.ExerciseSession`
- Offline evaluation: `scripts/evaluate_pushup.py` / `scripts/evaluate_squat.py` -> `core.evaluate_labeled_videos`
- Camera demo: `scripts/pushup_counter.py` / `scripts/squat_counter.py` -> `core.ExerciseSession`

## Quick start

```bash
pip install -r requirements.txt
python scripts/check.py
python run.py
```

Web app default: `http://localhost:5000`

## Script utilitas

```bash
python scripts/check.py
python scripts/dataset_audit.py
python scripts/evaluate_pushup.py
python scripts/evaluate_squat.py
python scripts/pushup_counter.py
python scripts/squat_counter.py
```

## Catatan teknis

- Source of truth runtime/evaluasi ada di `core/engine.py`, `core/video.py`, dan `config.py`.
- `artifacts/dataset_manifest.csv` dan `artifacts/dataset_issues.json` dihasilkan dari `scripts/dataset_audit.py`.
- Website saat ini masih backend-assisted: browser mengirim frame ke Flask untuk diproses per frame.
- Untuk publish ke Hugging Face saat ini yang realistis adalah repo pipeline/dataset/Space, bukan checkpoint model custom.
