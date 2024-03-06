# アプリケーションのルート（URLパターン）とそれに対応するビュー関数を定義

from myapp import app
from flask import render_template, request
from .regist import *
from .auth import *

# サンプル
# @app.route("/hello")
# def hello():
#     app.logger.info('Hello endpoint was reached')
#     return "Hello World!"


# トップ
@app.route("/")
def index():
    return render_template('index.html', page_title="トップ")

# 登録ページ
@app.route("/regist")
def regist():
    return render_template('rhythm/regist.html', page_title="登録")
# データ登録API
@app.route("/post_regist_data", methods=['POST'])
def post_regist_data():
    data = request.get_json()
    return save_request_data(data)

# 認証ページ
@app.route("/auth")
def auth():
    return render_template('rhythm/auth.html', page_title="認証")
# データ認証API
@app.route("/post_auth_data", methods=['POST'])
def post_auth_data():
    data = request.get_json()
    return auth_request_data(data)



# トップ(デモ)
@app.route("/demo")
def demo():
    return render_template('demo/top.html', page_title="デモトップ")

# 加速度センサー(デモ)
@app.route("/demo/acceleration")
def demo_acceleration():
    return render_template('demo/acceleration.html', page_title="加速度センサーデモ")

# ジャイロセンサー(デモ)
@app.route("/demo/gyro")
def demo_gyro():
    return render_template('demo/gyro.html', page_title="ジャイロセンサーデモ")

# 磁気センサー(デモ)
@app.route("/demo/magnetism")
def demo_magnetism():
    return render_template('demo/magnetism.html', page_title="磁気センサーデモ")

# バイブレーション(デモ)
@app.route("/demo/vibration")
def demo_vibration():
    return render_template('demo/vibration.html', page_title="バイブレーションデモ")

# 音声出力(デモ)
@app.route("/demo/audio_output")
def demo_audio_output():
    return render_template('demo/audio_output.html', page_title="音声出力デモ")
