import sys
sys.path.append('../')
from myapp.svm import *
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier##挿入
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pickle
import argparse
import datetime
import glob


dir_path = "../data"
tmp2_dir_path = "./tmp2/"

test_name = "user1" # 検証したい人の名前
test_id = 1 # 検証したい人のid 

#------------------以下，モデル読み込み→認証（別プログラムでの実行ができます）----------------------------------------------    
   #モデル読み込み＝＞認証計算

input_file = tmp2_dir_path + 'output.csv'
column_index = 0  # 読み取る列のインデックス

data_set = set()  # 重複を許さないデータの集合

with open(input_file, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[column_index]:  # 空欄でない場合のみデータを追加
            data = row[column_index]
            non_numeric_data = re.sub(r'\d+', '', data)  # 数字以外の文字列を削除
            non_numeric_data = non_numeric_data.strip()  # 文字列の前後の空白を削除
            if non_numeric_data:  # 空文字列でない場合のみ追加
                data_set.add(non_numeric_data)

undata_list = list(data_set)  # リストに変換
# "Unnamed"という要素を削除してリストを更新
data_list = [item for item in undata_list if item != "Unnamed:"]

# print(data_list)
eer_threshold = {}
model_dir_name = '../data_svm_model2/'
data_auth_dir_name = '../data_auth2/'

for index, data_name in enumerate(data_list): # index: インデックス番号、data: リストの要素

    # modelを読み込み
    modelfile = model_dir_name+'binary-SVM-'+data_name+'-rhythm.sav'
    with open(modelfile, mode='rb') as f:
        model = pickle.load(f)
        print(modelfile)

    # scalerを読み込み
    scfile = f'{model_dir_name}binary-sc-{data_name}-rhythm.pkl'
    sc = pickle.load(open(scfile, "rb"))
    print("sc=>",scfile)

    # 特徴量データを入力できるように整形
    extracted_csv_path = data_auth_dir_name + '20230720_082107_user2_extracted_features.csv'
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
    # print(x_std)

    # 検証用データで精度を確認
    # pred_test = model.predict(X_test_std)
    prob = model.predict_proba(x_std)[:, 1]

    # print(type(prob))

    # print(type(eer_threshold))

    # print(prob)

    # print(eer_threshold)
    
    input_file = model_dir_name + 'Ans_SVM_{}.csv'.format(data_name)
    row_index = 1  # 行番号（0から始まる）
    column_index = 3  # 列番号（0から始まる）

    eer_threshold = None

    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == row_index:
                if column_index < len(row):
                    eer_threshold = row[column_index]
                break

    eer_threshold = float(eer_threshold)
    print(eer_threshold)
    for p in prob:
        print(p)
        if p > eer_threshold:
            pred_test = 1
        else:
            pred_test = 0

    print('predict ==> '+ str(pred_test))



    