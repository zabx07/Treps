"""Standalone Flask app runner."""

from app import create_app
from app.config import get_config


app = create_app(config_class=get_config())


if __name__ == "__main__":
    host = app.config.get("HOST", "0.0.0.0")
    port = app.config.get("PORT", 5000)
    debug = app.config.get("DEBUG", True)
    app.run(debug=debug, host=host, port=port)
