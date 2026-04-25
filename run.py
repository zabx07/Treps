#!/usr/bin/env python
"""Stable Flask app runner for Treps."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app
from app.config import get_config


app = create_app(config_class=get_config())


if __name__ == "__main__":
    host = app.config.get("HOST", "0.0.0.0")
    port = app.config.get("PORT", 5000)
    print("=" * 60)
    print("  TREPS FITNESS DETECTION APP")
    print("=" * 60)
    print(f"Project Root : {PROJECT_ROOT}")
    print(f"Run URL      : http://127.0.0.1:{port}")
    print(f"Network URL  : http://localhost:{port}")
    print("\nPress CTRL+C to stop\n")
    app.run(
        debug=False,
        host=host,
        port=port,
        use_reloader=False,
        threaded=True,
    )
