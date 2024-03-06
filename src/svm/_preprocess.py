####################################################################################################
# データの前処理
####################################################################################################


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
tmp_dir_path = "./tmp/"
id_name_dict = {}

if __name__=='__main__':
    
    ### データ取得 & 整形
    df_list = getShapingAllData(dir_path)

    ### 特徴量抽出
    for idx, (person_name, df_person) in enumerate(df_list.items()):

        # 特徴量抽出
        extracted_features_person = dataframe2feature(df_person)

        # CSV出力 (debug用)
        # print(extracted_features_person)
        extracted_features_person.to_csv(f'{tmp_dir_path}extracted_features_{idx+1}.csv', encoding='cp932')

        # IDと名前の関係保存
        id_name_dict[idx+1] = person_name

    # CSV結合 & ラベル付け
    # 結合するCSVファイルのパス
    csv_files = ['extracted_features_1.csv', 'extracted_features_2.csv']
    target_file = 'extracted_features_2.csv'  # ラベル1を付与する対象のファイル名
    target_word = '2'  # 指定したワード

    # 既存のファイルにfile_name列を追加し、ラベル列を初期化
    for file in csv_files:
        data = pd.read_csv(file)
        data.insert(0, 'file_name', file)  # ファイル名を追加
        data['label'] = 0  # ラベル列を初期化
        data.to_csv(file, index=False)

    # 結合したデータを格納するための空のデータフレームを作成
    combined_data = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

    # 指定したワードを含むファイルにはラベル1を設定する
    combined_data.loc[combined_data['file_name'].str.contains(target_word), 'label'] = 1

    # 結合したデータを保存する
    combined_data.to_csv('combined_data_2.csv', index=False)