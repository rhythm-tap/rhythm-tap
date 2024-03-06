####################################################################################################
# 登録フェーズ
####################################################################################################

from flask import jsonify
from myapp import app
import os
import json
import csv
from datetime import datetime

def save_request_data(data):
    request = {}
    if not data:
        request['result'] = False
        request['message'] = "データ正常に送信されませんでした。"
        return jsonify(request), 400
    dir_name = './data/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    time_format = getStrTime(data['start_time'])
    # dump_json(data, '{dir_name}{name}_{start_time}.json'.format(dir_name=dir_name, name=data['name'], start_time=time_format))
    dump_csv(data, '{dir_name}{name}_{start_time}.csv'.format(dir_name=dir_name, name=data['name'], start_time=time_format))
    request['result'] = True
    request['message'] = "データ正常に保存されました。"
    return jsonify(request), 200


def dump_json(data, file_name='./data/data.json'):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def dump_csv(data, file_name='./data/data.csv'):
    with open(file_name, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data['regist_data'])

def getStrTime(timestamp):
    return datetime.fromtimestamp(timestamp / 1000).strftime("%Y%m%d_%H%M%S")
