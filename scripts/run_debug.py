#!/usr/bin/env python
"""Run the Flask app with verbose logging."""

import logging
import os
import sys
from pathlib import Path


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTHONPATH"] = str(PROJECT_ROOT)


if __name__ == "__main__":
    try:
        from app import create_app
        from app.config import get_config

        app = create_app(config_class=get_config())
        logger.info("Project Root: %s", PROJECT_ROOT)
        logger.info("App Directory: %s", APP_DIR)
        logger.info("Template folder: %s", app.template_folder)
        logger.info("Static folder: %s", app.static_folder)

        app.run(
            debug=True,
            host="0.0.0.0",
            port=5000,
            use_reloader=True,
            use_debugger=True,
            threaded=True,
        )
    except Exception as exc:
        logger.error("Failed to start app: %s", exc, exc_info=True)
        print(f"ERROR: {exc}")
        sys.exit(1)
