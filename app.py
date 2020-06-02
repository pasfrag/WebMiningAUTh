from flask import Flask

def create_app():
    from views import main
    app = Flask(__name__)

    app.register_blueprint(main)

    app.run()


if __name__ == '__main__':
    create_app()
