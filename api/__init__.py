"""
SENTINEL API Package

Flask application factory and route registration.
"""

import threading
from flask import Flask

from config import get_config


def create_app() -> Flask:
    """
    Create and configure Flask application.
    
    Returns:
        Configured Flask app instance
    """
    cfg = get_config()
    
    app = Flask(
        __name__,
        template_folder='../templates',
        static_folder='../static',
    )
    
    # Configure app
    app.config['SECRET_KEY'] = 'sentinel-defence-platform'
    app.config['JSON_SORT_KEYS'] = False
    
    # Register routes
    from api.routes import bp, init_session, db_writer_loop
    app.register_blueprint(bp)
    
    # Initialize session
    init_session()
    
    # Start DB writer thread
    db_thread = threading.Thread(target=db_writer_loop, daemon=True)
    db_thread.start()
    
    return app


__all__ = ["create_app"]
