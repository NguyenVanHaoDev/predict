from app import create_app
from app.config import Settings

app = create_app()

if __name__ == '__main__':
    app.run(debug=Settings.DEBUG)