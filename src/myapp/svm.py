####################################################################################################
# SVM・特徴量抽出
####################################################################################################




import os
import glob
import re
import pandas as pd
from tsfresh.feature_extraction import extract_features



### CSVファイルからデータ取得 & 整形 ###
def getShapingData(csv_path, person_name=None, idx=0):

    # ファイル名から人物名を取得
    if person_name is None:
        csv_match = re.search(r'^(.*)_\d{8}_\d{6}\.csv$', os.path.basename(csv_path))
        if csv_match is None:
            return None
        person_name = csv_match[1]
    # CSVファイルを読み込む
    df = pd.read_csv(csv_path, header=None, encoding='utf-8')
    # 必要なカラムを追加
    df['id'] = person_name + str(idx)
    df['time'] = df.index+1
    # データフレームのカラム名を設定
    df.columns = ['time_stamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_alpha', 'gyro_beta', 'gyro_gamma', 'flag', 'id', 'time']
    # 必要なカラムのみを取り出す
    df = df[['id', 'time', 'acc_x', 'acc_y', 'acc_z', 'gyro_alpha', 'gyro_beta', 'gyro_gamma']]

    # CSV出力 (debug用)
    # df.to_csv(f'./output_person_single.csv', index=False, encoding='utf-8')

    return df


### CSVデータ取得 & 整形 ###
def getShapingAllData(dir_path):

    csv_path_list = sorted(glob.glob(dir_path+"/*.csv"))

    df_list = {}
    df_list_per_person = []
    past_person_name = None
    idx = 1
    for file_path in csv_path_list:

        # ファイル名から人物名を取得
        csv_match = re.search(r'^(.*)_\d{8}_\d{6}\.csv$', os.path.basename(file_path))
        if csv_match is None:
            continue
        person_name = csv_match[1]
        if past_person_name == person_name:
            idx += 1
        else:
            idx = 1

        # dataframe取得
        df = getShapingData(csv_path=file_path, person_name=person_name, idx=idx)
        if df is None:
            continue

        # 人ごとにdataframeを作成
        if past_person_name == person_name:
            df_list_per_person.append(df)
        elif past_person_name is None:
            # 最初
            df_list_per_person.append(df)
        elif df_list_per_person != []:
            df_person = pd.concat(df_list_per_person, ignore_index=True)
            df_list[past_person_name] = df_person
            df_list_per_person = []
            df_list_per_person.append(df)
        past_person_name = person_name

    # 最後のdataframe作成
    df_person = pd.concat(df_list_per_person, ignore_index=True)
    df_list[past_person_name] = df_person
    df_list_per_person = []
    df_list_per_person.append(df)

    # 全てのデータフレームを結合してCSV出力 (debug用)
    # for i, (person_name, p_df) in enumerate(df_list.items()):
    #     p_df.to_csv(f'./output_{person_name}.csv', index=False, encoding='utf-8')
    # total_df = pd.concat(df_list.values(), ignore_index=True)
    # total_df.to_csv(f'./output_total.csv', index=False, encoding='utf-8')

    return df_list


### dataframe -> 特徴量抽出 ###
def dataframe2feature(df):
    # 生成する特徴量を指定
    fc_parameters = {
        'median': None,
        'mean': None,
        'length': None,
        'standard_deviation': None,
        'last_location_of_maximum': None,
        'first_location_of_maximum': None,
        'last_location_of_minimum': None,
        'first_location_of_minimum': None,
        'sample_entropy': None,
        'maximum': None,
        'absolute_maximum': None,
        'minimum': None,
        'autocorrelation': [{'lag': 0},
        {'lag': 1},
        {'lag': 2},
        {'lag': 3},
        {'lag': 4},
        {'lag': 5},
        {'lag': 6},
        {'lag': 7},
        {'lag': 8},
        {'lag': 9}],
        'number_peaks': [{'n': 1},
        {'n': 3},
        {'n': 5},
        {'n': 10},
        {'n': 50}]
    }

    extracted_features = extract_features(
        timeseries_container=df,
        default_fc_parameters=fc_parameters,
        column_id='id',
        column_sort='time',
        column_kind=None,
        column_value=None
    )

    # CSV出力 (debug用)
    # extracted_features.to_csv(f'extracted_features.csv', encoding='utf-8')

    return extracted_features