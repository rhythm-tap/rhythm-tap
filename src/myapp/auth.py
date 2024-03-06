####################################################################################################
# 認証フェーズ
####################################################################################################

from flask import jsonify
from myapp import app
from .dtw import *
from .svm import *
from .dl import *
from .regist import getStrTime
import os
import re
import csv
import pandas as pd
import pickle
from scipy.signal import find_peaks


def auth_request_data(data):
    request = {}
    if not data:
        request['result'] = False
        request['message'] = "データ正常に送信されませんでした。"
        return jsonify(request), 400

    res_bool, res = auth_dl(data)  # ここを変更すれば認証方法を変更できる
    if not res_bool:
        request['result'] = False
        request['message'] = res
        return jsonify(request), 500

    request['result'] = True
    request['auth_result'] = res
    request['message'] = "正常に認証処理できました。"
    return jsonify(request), 200


dtw_func = linear_dtw
# dtw_func = nonlinear_dtw
def auth_dtw(data):
    dir_name = './data/'
    auth_name = data['name']
    # 認証の準備ができているかどうか（閾値データがあるか）
    if not os.path.exists(dir_name+auth_name+'.txt'):
        return False, "認証の準備ができていません。"
    # 閾値データの読み込み
    with open(dir_name+auth_name+'.txt', 'r') as f:
        thres_data = f.readlines()[-1].split(', ')
        thres_path = thres_data[0]
        thresholds = thres_data[1:]
        thresholds = [float(element) if element != 'None' else None for element in thresholds]
    # 基準データの読み込み
    if not os.path.exists(dir_name+thres_path):
        return False, "基準ファイルがありません。"
    data_origin = read_csv(dir_name+thres_path)
    # postデータ
    data_post = data['auth_data']
    # データの整形
    data_post = [list(row) for row in zip(*data_post)]
    data_post = [[float(element) if is_number(element) else element for element in row] for row in data_post]
    data_origin = [list(row) for row in zip(*data_origin)]
    data_origin = [[float(element) if is_number(element) else element for element in row] for row in data_origin]
    # DTW算出
    check_idxs = [0,1,2]
    idx_table = {0:"acc_x", 1:"acc_y", 2:"acc_z", 3:"gyro_alpha", 4:"gyro_beta", 5:"gyro_gamma"}
    res = {"result": True, "auth_name": auth_name}
    for idx in check_idxs:
        distance = dtw_func(data_origin[idx+1], data_post[idx+1])
        res[str(idx)] = idx_table[idx] + ": " + str(distance)
        if thresholds[idx] is None:
            continue
        if thresholds[idx] <= distance:
            res["result"] = False
    return True, res


def auth_svm(data):
    feature_pattern = 1
    if feature_pattern == 1:
        tmp_dir_name = './data_auth/'
        model_dir_name = './data_svm_model/'
    elif feature_pattern == 2:
        tmp_dir_name = './data_auth2/'
        model_dir_name = './data_svm_model2/'
    auth_name = data['name']
    time_format = getStrTime(data['start_time'])

    # 認証の準備ができているかどうか（閾値データがあるか）
    modelfile = model_dir_name+'binary-SVM-'+auth_name+'-rhythm.sav'
    if not os.path.exists(modelfile):
        return False, "認証の準備ができていません。"

    # リアルタイムに取得したデータをCSV保存
    if not os.path.exists(tmp_dir_name):
        os.mkdir(tmp_dir_name)
    csv_path = '{dir_name}{name}_{start_time}.csv'.format(dir_name=tmp_dir_name, name=data['name'], start_time=time_format)
    dump_csv(data, csv_path)

    # 保存したCSVから読み込んでデータ整形
    df = getShapingData(csv_path, person_name=data['name'])
    
    # 取得したデータから特徴量を抽出し、CSV保存
    extracted_features = dataframe2feature(df)
    extracted_csv_path = f'{tmp_dir_name}{time_format}_{auth_name}_tsfresh_features.csv'
    extracted_features.to_csv(extracted_csv_path, encoding='utf-8')


    # CSVファイルをDataFrameとして読み込む
    data_frame = pd.read_csv(csv_path)
    
    # 先頭行のTimestampを抜き出す
    first_timestamp = int(data_frame.iloc[0, 0])
    # 最終行のTimestampを抜き出す
    last_timestamp = int(data_frame.iloc[-1, 0])
    # Timestampの差を計算する
    rhythm_time = last_timestamp - first_timestamp

    # マイナス方向のピークを検出
    data_z = data_frame.iloc[:, 3]
    peaks_z, _ = find_peaks(-data_z)
    negative_peaks_z = peaks_z[data_z.iloc[peaks_z] < -2]
    main_peaks_z = []
    peak_timestamps_z = []
    for peak in negative_peaks_z:
        left = peak - 1
        right = peak + 1
        if data_z.iloc[left] < data_z.iloc[peak] or data_z.iloc[right] < data_z.iloc[peak]:
            continue
        main_peaks_z.append(peak)
        peak_timestamps_z.append(data_frame.iloc[peak, 0])
    num_peaks_z = len(main_peaks_z)
    if len(main_peaks_z) > 0:
        max_peak_z = data_z.iloc[main_peaks_z].min()
    else:
        max_peak_z = 0
    
    # 側面を叩いた場合
    data_x = data_frame.iloc[:, 1]
    peaks_x, _ = find_peaks(-data_x)
    negative_peaks_x = peaks_x[data_x.iloc[peaks_x] < -2]
    main_peaks_x = []
    peak_timestamps_x = []
    for peak in negative_peaks_x:
        left = peak - 1
        right = peak + 1
        if data_x.iloc[left] < data_x.iloc[peak] or data_x.iloc[right] < data_x.iloc[peak]:
            continue
        main_peaks_x.append(peak)
        peak_timestamps_x.append(data_frame.iloc[peak, 0])
    num_peaks_x = len(main_peaks_x)
    if len(main_peaks_x) > 0:
        max_peak_x = data_x.iloc[main_peaks_x].min()
    else:
        max_peak_x = 0
    
    # 上面を叩いた場合
    data_y = data_frame.iloc[:, 2]
    peaks_y, _ = find_peaks(-data_y)
    negative_peaks_y = peaks_y[data_y.iloc[peaks_y] < -2]
    main_peaks_y = []
    peak_timestamps_y = []
    for peak in negative_peaks_y:
        left = peak - 1
        right = peak + 1
        if data_y.iloc[left] < data_y.iloc[peak] or data_y.iloc[right] < data_y.iloc[peak]:
            continue
        main_peaks_y.append(peak)
        peak_timestamps_y.append(data_frame.iloc[peak, 0])
    num_peaks_y = len(main_peaks_y)
    if len(main_peaks_y) > 0:
        max_peak_y = data_y.iloc[main_peaks_y].min()
    else:
        max_peak_y = 0
    
    # タップエネルギー
    tap_energy = max_peak_x**2 + max_peak_y**2 + max_peak_z**2

    # カラム名とデータのリスト
    column_names = ["rhythm_time","num_peaks_x","num_peaks_y","num_peaks_z","max_peak_x","max_peak_y","max_peak_z","tap_energy"]
    rhythm_data = [rhythm_time,num_peaks_x,num_peaks_y,num_peaks_z,max_peak_x,max_peak_y,max_peak_z,tap_energy]

    csv_file_path = f'{tmp_dir_name}{time_format}_{auth_name}_rhythm_features.csv'
    # CSVファイルにデータを書き込む
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)  # カラム名を書き込む
        writer.writerow(rhythm_data)  # データを書き込む
    
    file1 = f'{tmp_dir_name}{time_format}_{auth_name}_tsfresh_features.csv'
    file2 = f'{tmp_dir_name}{time_format}_{auth_name}_rhythm_features.csv'

    # CSVファイルをDataFrameとして読み込む
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # ファイル1とファイル2を結合する
    merged_df = pd.concat([df1, df2], axis=1)

    # 結合したDataFrameをCSVファイルとして保存
    merged_df.to_csv(f'{tmp_dir_name}{time_format}_{auth_name}_extracted_features.csv', index=False)
    
   
    # modelを読み込み
    with open(modelfile, mode='rb') as f:
        model = pickle.load(f)
        # print(modelfile)

    # scalerを読み込み
    sc = pickle.load(open(f'{model_dir_name}binary-sc-{auth_name}-rhythm.pkl', "rb"))

    # 特徴量データを入力できるように整形
    input_data = pd.read_csv(extracted_csv_path, sep=",", encoding='utf-8')
    x = input_data.loc[:, ['acc_x__median',
    'acc_x__mean', 
    'acc_x__length', 
    'acc_x__standard_deviation', 
    'acc_x__last_location_of_maximum', 
    'acc_x__first_location_of_maximum', 
    'acc_x__last_location_of_minimum',
    'acc_x__first_location_of_minimum', 
    'acc_x__sample_entropy', 
    'acc_x__maximum',
    'acc_x__absolute_maximum', 
    'acc_x__minimum', 
    'acc_x__autocorrelation__lag_0', 
    'acc_x__autocorrelation__lag_1', 
    'acc_x__autocorrelation__lag_2', 
    'acc_x__autocorrelation__lag_3',
    'acc_x__autocorrelation__lag_4', 
    'acc_x__autocorrelation__lag_5', 
    'acc_x__autocorrelation__lag_6',
    'acc_x__autocorrelation__lag_7', 
    'acc_x__autocorrelation__lag_8', 
    'acc_x__autocorrelation__lag_9',
    'acc_x__number_peaks__n_1', 
    'acc_x__number_peaks__n_3', 
    'acc_x__number_peaks__n_5',
    'acc_x__number_peaks__n_10', 
    'acc_x__number_peaks__n_50',
    'acc_y__median',
    'acc_y__mean', 
    'acc_y__length', 
    'acc_y__standard_deviation', 
    'acc_y__last_location_of_maximum', 
    'acc_y__first_location_of_maximum', 
    'acc_y__last_location_of_minimum',
    'acc_y__first_location_of_minimum', 
    'acc_y__sample_entropy', 
    'acc_y__maximum',
    'acc_y__absolute_maximum', 
    'acc_y__minimum', 
    'acc_y__autocorrelation__lag_0', 
    'acc_y__autocorrelation__lag_1', 
    'acc_y__autocorrelation__lag_2', 
    'acc_y__autocorrelation__lag_3',
    'acc_y__autocorrelation__lag_4', 
    'acc_y__autocorrelation__lag_5', 
    'acc_y__autocorrelation__lag_6',
    'acc_y__autocorrelation__lag_7', 
    'acc_y__autocorrelation__lag_8', 
    'acc_y__autocorrelation__lag_9',
    'acc_y__number_peaks__n_1', 
    'acc_y__number_peaks__n_3', 
    'acc_y__number_peaks__n_5',
    'acc_y__number_peaks__n_10', 
    'acc_y__number_peaks__n_50',
    'acc_z__median',
    'acc_z__mean', 
    'acc_z__length', 
    'acc_z__standard_deviation', 
    'acc_z__last_location_of_maximum', 
    'acc_z__first_location_of_maximum', 
    'acc_z__last_location_of_minimum',
    'acc_z__first_location_of_minimum', 
    'acc_z__sample_entropy', 
    'acc_z__maximum',
    'acc_z__absolute_maximum', 
    'acc_z__minimum', 
    'acc_z__autocorrelation__lag_0', 
    'acc_z__autocorrelation__lag_1', 
    'acc_z__autocorrelation__lag_2', 
    'acc_z__autocorrelation__lag_3',
    'acc_z__autocorrelation__lag_4', 
    'acc_z__autocorrelation__lag_5', 
    'acc_z__autocorrelation__lag_6',
    'acc_z__autocorrelation__lag_7', 
    'acc_z__autocorrelation__lag_8', 
    'acc_z__autocorrelation__lag_9',
    'acc_z__number_peaks__n_1', 
    'acc_z__number_peaks__n_3', 
    'acc_z__number_peaks__n_5',
    'acc_z__number_peaks__n_10', 
    'acc_z__number_peaks__n_50',
    'gyro_alpha__median',
    'gyro_alpha__mean', 
    'gyro_alpha__length', 
    'gyro_alpha__standard_deviation', 
    'gyro_alpha__last_location_of_maximum', 
    'gyro_alpha__first_location_of_maximum', 
    'gyro_alpha__last_location_of_minimum',
    'gyro_alpha__first_location_of_minimum', 
    'gyro_alpha__sample_entropy', 
    'gyro_alpha__maximum',
    'gyro_alpha__absolute_maximum', 
    'gyro_alpha__minimum', 
    'gyro_alpha__autocorrelation__lag_0', 
    'gyro_alpha__autocorrelation__lag_1', 
    'gyro_alpha__autocorrelation__lag_2', 
    'gyro_alpha__autocorrelation__lag_3',
    'gyro_alpha__autocorrelation__lag_4', 
    'gyro_alpha__autocorrelation__lag_5', 
    'gyro_alpha__autocorrelation__lag_6',
    'gyro_alpha__autocorrelation__lag_7', 
    'gyro_alpha__autocorrelation__lag_8', 
    'gyro_alpha__autocorrelation__lag_9',
    'gyro_alpha__number_peaks__n_1', 
    'gyro_alpha__number_peaks__n_3', 
    'gyro_alpha__number_peaks__n_5',
    'gyro_alpha__number_peaks__n_10', 
    'gyro_alpha__number_peaks__n_50',
    'gyro_beta__median',
    'gyro_beta__mean', 
    'gyro_beta__length', 
    'gyro_beta__standard_deviation', 
    'gyro_beta__last_location_of_maximum', 
    'gyro_beta__first_location_of_maximum', 
    'gyro_beta__last_location_of_minimum',
    'gyro_beta__first_location_of_minimum', 
    'gyro_beta__sample_entropy', 
    'gyro_beta__maximum',
    'gyro_beta__absolute_maximum', 
    'gyro_beta__minimum', 
    'gyro_beta__autocorrelation__lag_0', 
    'gyro_beta__autocorrelation__lag_1', 
    'gyro_beta__autocorrelation__lag_2', 
    'gyro_beta__autocorrelation__lag_3',
    'gyro_beta__autocorrelation__lag_4', 
    'gyro_beta__autocorrelation__lag_5', 
    'gyro_beta__autocorrelation__lag_6',
    'gyro_beta__autocorrelation__lag_7', 
    'gyro_beta__autocorrelation__lag_8', 
    'gyro_beta__autocorrelation__lag_9',
    'gyro_beta__number_peaks__n_1', 
    'gyro_beta__number_peaks__n_3', 
    'gyro_beta__number_peaks__n_5',
    'gyro_beta__number_peaks__n_10', 
    'gyro_beta__number_peaks__n_50',
    'gyro_gamma__median',
    'gyro_gamma__mean', 
    'gyro_gamma__length', 
    'gyro_gamma__standard_deviation', 
    'gyro_gamma__last_location_of_maximum', 
    'gyro_gamma__first_location_of_maximum', 
    'gyro_gamma__last_location_of_minimum',
    'gyro_gamma__first_location_of_minimum', 
    'gyro_gamma__sample_entropy', 
    'gyro_gamma__maximum',
    'gyro_gamma__absolute_maximum', 
    'gyro_gamma__minimum', 
    'gyro_gamma__autocorrelation__lag_0', 
    'gyro_gamma__autocorrelation__lag_1', 
    'gyro_gamma__autocorrelation__lag_2', 
    'gyro_gamma__autocorrelation__lag_3',
    'gyro_gamma__autocorrelation__lag_4', 
    'gyro_gamma__autocorrelation__lag_5', 
    'gyro_gamma__autocorrelation__lag_6',
    'gyro_gamma__autocorrelation__lag_7', 
    'gyro_gamma__autocorrelation__lag_8', 
    'gyro_gamma__autocorrelation__lag_9',
    'gyro_gamma__number_peaks__n_1', 
    'gyro_gamma__number_peaks__n_3', 
    'gyro_gamma__number_peaks__n_5',
    'gyro_gamma__number_peaks__n_10', 
    'gyro_gamma__number_peaks__n_50',
    'rhythm_time',
    'num_peaks_x', 
    'num_peaks_y', 
    'num_peaks_z', 
    'max_peak_x', 
    'max_peak_y', 
    'max_peak_z', 
    'tap_energy'] ].values
    x_std = sc.transform(x)#xはリアルタイムで取得したデータです(特徴抽出後)

    # 精度を確認
    prob = model.predict_proba(x_std)[0, 1]

    # 閾値よりも高いかどうか
    input_file = model_dir_name+'Ans_SVM_{}.csv'.format(auth_name)
    row_index = 1  # 行番号（0から始まる）
    column_index = 3  # 列番号（0から始まる）
    eer_threshold = None
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == row_index:
                if column_index < len(row):
                    eer_threshold = row[column_index]
                break
    eer_threshold = float(eer_threshold)
    auth_result = True if prob > eer_threshold else False

    res = {"result": auth_result, "auth_name": auth_name, "0": "output: "+str(prob), "1": "eer_threshold: "+str(eer_threshold)}
    return True, res


def auth_dl(data):
    tmp_dir_name = './data_auth_dl/'
    model_dir_name = './deeplearning/log_sample/'
    auth_name = data['name']
    time_format = getStrTime(data['start_time'])

    # 認証の準備ができているかどうか（閾値データがあるか）
    pth_dirs = search_files(model_dir_name, rf".*\/log_{auth_name}_\d{{8}}_\d{{6}}\/model_checkpoint\.pth$")
    if pth_dirs is None or len(pth_dirs) != 1:
        return False, "認証の準備ができていません。"
    model_path = pth_dirs[0]

    # リアルタイムに取得したデータをCSV保存
    if not os.path.exists(tmp_dir_name):
        os.mkdir(tmp_dir_name)
    csv_path = '{dir_name}{name}_{start_time}.csv'.format(dir_name=tmp_dir_name, name=data['name'], start_time=time_format)
    dump_csv(data, csv_path)

    # テスト
    prob = test_deeplearning(model_path, csv_path)
    auth_result = True if prob>=0.5 else False

    # テストしたログを保存（追記）
    log_csv_path = '{dir_name}{name}.csv'.format(dir_name=tmp_dir_name, name=data['name'])
    if not os.path.exists(log_csv_path):
        with open(log_csv_path, 'w') as file:
            pass
    log_data = read_csv(log_csv_path)
    log_data.append([time_format, prob])
    dump_csv({"auth_data": log_data}, log_csv_path)

    res = {"result": auth_result, "auth_name": auth_name, "0": "output: "+str(prob)}
    return True, res


def read_csv(csv_path):
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            data.append(row)
    return data

def dump_csv(data, file_name='./data/data.csv'):
    with open(file_name, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data['auth_data'])

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# ディレクトリ中のファイルを検索する関数
def search_files(directory, pattern):
    result_list = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if re.search(pattern, os.path.join(dirpath, filename)):
                result_list.append(os.path.join(dirpath, filename))
    if result_list == []:
        return None
    return result_list