from flask import render_template, Blueprint
from json_parser import VisualizationParsing

main = Blueprint('main', __name__)

@main.route('/')
@main.route('/index')
def index():
    parser = VisualizationParsing()
    data = parser.finalize_data()
    return render_template('index.html', data=data)
