from flask import Blueprint

main_bp = Blueprint(
    'main', __name__,
    static_folder='static',
    static_url_path='/main_static',
    template_folder='templates'
)

from . import routes  # import routes.py trong cùng thư mục
