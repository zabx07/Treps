# Flask app package
# Initialize Flask application

from flask import Flask
from pathlib import Path
import os
import config as project_config

def create_app(config_class=None):
    """Application factory function"""
    CURRENT_DIR = Path(__file__).parent
    TEMPLATE_DIR = CURRENT_DIR / 'templates'
    STATIC_DIR = CURRENT_DIR / 'static'

    # Create static directory if it doesn't exist
    STATIC_DIR.mkdir(exist_ok=True)

    app = Flask(
        __name__,
        template_folder=str(TEMPLATE_DIR),
        static_folder=str(STATIC_DIR),
        static_url_path='/static'
    )

    # Load configuration
    if config_class:
        app.config.from_object(config_class)
    else:
        # Default config
        app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'treps_fitness_secret_2024')
        app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    app.config["MAX_CONTENT_LENGTH"] = (
        project_config.SERVER_UPLOAD_FALLBACK_MAX_FILE_MB * 1024 * 1024
    )
    app.config["JSON_SORT_KEYS"] = False

    # Import and register blueprints/routes here if needed
    # For now, we'll keep routes in main.py

    # Register blueprint
    from .routes import main_bp
    app.register_blueprint(main_bp)

    return app

