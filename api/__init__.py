"""
SENTINEL API Package

Flask application factory and route registration.
"""

import logging
import threading
from flask import Flask

from config import get_config

logger = logging.getLogger(__name__)


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
    from api.routes import bp, init_session, db_writer_loop, start_detection_worker
    app.register_blueprint(bp)
    
    # Initialize session (sets session_id)
    init_session()
    
    # Start DB writer thread
    db_thread = threading.Thread(target=db_writer_loop, daemon=True)
    db_thread.start()
    
    # Start detection worker (after session_id is set)
    start_detection_worker()
    
    return app


__all__ = ["create_app"]
