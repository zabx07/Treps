#!/usr/bin/env python
"""Runtime pre-flight check for Treps web app and upload fallback."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_ok(message: str):
    print(f"    OK  {message}")


def print_fail(message: str):
    print(f"    ERR {message}")


def print_warn(message: str):
    print(f"    WARN {message}")


def recommended_python_command() -> str:
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
      return ".\\.venv\\Scripts\\python.exe"
    return "python"


def main():
    python_cmd = recommended_python_command()

    print("=" * 60)
    print("  TREPS FITNESS DETECTION - PRE-FLIGHT CHECK")
    print("=" * 60)

    print("\n[1] Python Version Check")
    print(f"    Python: {sys.version}")
    if sys.version_info < (3, 10):
        print_fail("Python 3.10+ required")
        raise SystemExit(1)
    print_ok("Python version")

    print("\n[2] Directory Structure Check")
    required_dirs = [
        PROJECT_ROOT / "app",
        PROJECT_ROOT / "app/templates",
        PROJECT_ROOT / "core",
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT / "scripts",
    ]
    for dir_name in required_dirs:
        if not dir_name.exists():
            print_fail(f"{dir_name.relative_to(PROJECT_ROOT)}/ NOT FOUND")
            raise SystemExit(1)
        print_ok(f"{dir_name.relative_to(PROJECT_ROOT)}/ exists")

    print("\n[3] Required Files Check")
    required_files = [
        PROJECT_ROOT / "app/main.py",
        PROJECT_ROOT / "app/__init__.py",
        PROJECT_ROOT / "app/routes.py",
        PROJECT_ROOT / "app/templates/home.html",
        PROJECT_ROOT / "app/templates/tutorial.html",
        PROJECT_ROOT / "app/templates/exercise.html",
        PROJECT_ROOT / "app/static/js/exercise-runtime.js",
        PROJECT_ROOT / "core/engine.py",
        PROJECT_ROOT / "run.py",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "config.py",
    ]
    for file_name in required_files:
        if not file_name.exists():
            print_fail(f"{file_name.relative_to(PROJECT_ROOT)} NOT FOUND")
            raise SystemExit(1)
        print_ok(file_name.relative_to(PROJECT_ROOT))

    print("\n[4] Installed Packages Check")
    runtime_packages = {
        "flask": "Flask",
        "cv2": "OpenCV",
        "mediapipe": "MediaPipe",
        "numpy": "NumPy",
    }
    optional_packages = {
        "sklearn": "Scikit-learn",
        "matplotlib": "Matplotlib",
    }
    missing_runtime = []
    for import_name, display_name in runtime_packages.items():
        try:
            importlib.import_module(import_name)
            print_ok(display_name)
        except ImportError:
            print_fail(f"{display_name} NOT INSTALLED")
            missing_runtime.append(import_name)

    missing_optional = []
    for import_name, display_name in optional_packages.items():
        try:
            importlib.import_module(import_name)
            print_ok(f"{display_name} (optional eval)")
        except ImportError:
            print_warn(f"{display_name} missing - only needed for evaluation/artifacts")
            missing_optional.append(import_name)

    if missing_runtime:
        print(f"\n    Missing runtime packages: {', '.join(missing_runtime)}")
        print(f"    Install command: {python_cmd} -m pip install -r requirements.txt")
        raise SystemExit(1)

    print("\n[5] App Bootstrap Check")
    try:
        import config
        from app import create_app
        from app.config import get_config

        app = create_app(config_class=get_config())
        print_ok("Flask app loaded successfully")
        print_ok(f"Template folder: {app.template_folder}")
        print_ok(f"Static folder: {app.static_folder}")
        print_ok(
            f"Upload fallback route: {config.SERVER_UPLOAD_FALLBACK_ROUTE} "
            f"(enabled={config.SERVER_UPLOAD_FALLBACK_ENABLED})"
        )
    except Exception as exc:
        print_fail(f"Failed to load Flask app: {exc}")
        raise SystemExit(1)

    print("\n[6] Shared Engine Import Check")
    try:
        from core.engine import ExerciseSession

        ExerciseSession("push_up")
        ExerciseSession("squat")
        print_ok("Shared exercise engine")
    except Exception as exc:
        print_fail(f"Failed to initialize shared engine: {exc}")
        raise SystemExit(1)

    print("\n[7] Route / Template Rendering Check")
    try:
        client = app.test_client()
        expected_status = {
            "/": 200,
            "/health": 200,
            "/tutorial": 200,
            "/tutorial/squat": 200,
            "/tutorial/push_up": 200,
            "/exercise/squat": 200,
            "/exercise/push_up": 200,
        }
        for path, expected in expected_status.items():
            response = client.get(path)
            if response.status_code != expected:
                print_fail(f"{path} returned {response.status_code}, expected {expected}")
                raise SystemExit(1)
            print_ok(f"{path} -> {response.status_code}")

        response = client.post("/api/start_session", json={"exercise_type": "squat", "target_reps": 10})
        if response.status_code != 200:
            print_fail(f"/api/start_session returned {response.status_code}")
            raise SystemExit(1)
        print_ok("/api/start_session")

        response = client.post("/api/cleanup", json={"session_id": "smoke"})
        if response.status_code != 200:
            print_fail(f"/api/cleanup returned {response.status_code}")
            raise SystemExit(1)
        print_ok("/api/cleanup")

        response = client.post("/api/process_uploaded_video")
        if response.status_code not in {400, 413}:
            print_fail(f"/api/process_uploaded_video returned {response.status_code} on empty request")
            raise SystemExit(1)
        print_ok("/api/process_uploaded_video route available")
    except Exception as exc:
        print_fail(f"Route check failed: {exc}")
        raise SystemExit(1)

    print("\n" + "=" * 60)
    print("  PRE-FLIGHT CHECK PASSED")
    print("=" * 60)
    print("\nCommands:")
    print(f"  Install deps : {python_cmd} -m pip install -r requirements.txt")
    print(f"  Preflight    : {python_cmd} scripts/check.py")
    print(f"  Run web app  : {python_cmd} run.py")
    print(f"  Dataset audit: {python_cmd} scripts/dataset_audit.py")
    print(f"  Eval push-up : {python_cmd} scripts/evaluate_pushup.py")
    print(f"  Eval squat   : {python_cmd} scripts/evaluate_squat.py")
    print(f"  Live push-up : {python_cmd} scripts/pushup_counter.py")
    print(f"  Live squat   : {python_cmd} scripts/squat_counter.py")
    if missing_optional:
        print("\nOptional packages still missing for evaluation/artifacts:")
        print(f"  {', '.join(missing_optional)}")


if __name__ == "__main__":
    main()
