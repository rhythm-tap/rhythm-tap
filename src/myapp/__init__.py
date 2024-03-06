from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__, template_folder='./templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config.from_pyfile('config.py')

# DB接続
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://{user}:{password}@{host}/{name}'.format(user=app.config['DB_USER'], password=app.config['DB_PASSWORD'], host=app.config['DB_HOST_MASTER'], name=app.config['DB_NAME'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO']=True
db = SQLAlchemy(app)

# json文字化け対策
app.json.ensure_ascii = False
