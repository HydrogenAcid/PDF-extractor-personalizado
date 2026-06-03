from flask import send_from_directory

from src.utils.helpers import project_path


def register_welcome_assets(app):
    """Sirve recursos visuales de la pantalla inicial desde /assets."""

    @app.route("/assets/<path:filename>")
    def institutional_assets(filename):
        return send_from_directory(str(project_path("assets")), filename)
