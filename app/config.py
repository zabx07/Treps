"""
Flask Development Configuration
Konfigurasi untuk development server yang lebih stabil
"""
import os

class DevelopmentConfig:
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
    # Flask settings
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = True
    
    # Session settings
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour
    SESSION_TYPE = 'filesystem'
    
    # Server settings
    HOST = '0.0.0.0'
    PORT = 5000
    
    # Disable reloader on VSCode debugger
    # Set WERKZEUG_RUN_MAIN=true when running under debugger
    @staticmethod
    def get_reloader_setting():
        """
        Determine if reloader should be enabled.
        Returns False if running under debugger.
        """
        return os.environ.get('WERKZEUG_RUN_MAIN') != 'true'

class ProductionConfig:
    """Production configuration"""
    DEBUG = False
    TESTING = False
    JSON_SORT_KEYS = True

# Select configuration based on environment
def get_config():
    env = os.environ.get('FLASK_ENV', 'development')
    if env == 'production':
        return ProductionConfig()
    else:
        return DevelopmentConfig()
