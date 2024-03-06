####################################################################################################
# データの前処理
####################################################################################################


import sys
sys.path.append('../')
from myapp.svm import *
import csv
import glob
import os


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
        extracted_features_person.to_csv(f'{tmp_dir_path}extracted_features_{person_name}.csv', encoding='utf-8')

        # IDと名前の関係保存
        id_name_dict[idx+1] = person_name

    # CSV結合 & ラベル付け
    # まとめるファイルのパス
    output_file_path = tmp_dir_path + "input.csv"

    # # まとめるファイルのディレクトリパス
    # current_directory = os.getcwd()
    # directory_path = current_directory

    # # まとめるファイルの拡張子
    # file_extension = "*.csv"  # 読み込むファイルの拡張子を指定

    # ファイルをまとめるための初期化処理
    first_file = True

    # ディレクトリ内のファイルを順に処理
    for file_path in glob.glob(os.path.join(tmp_dir_path, "*.csv")):
        with open(file_path, "r", encoding='utf-8') as file:
            # ファイルの内容を読み込む
            file_data = file.read().strip()

            # 1行目のカラムを追加する
            if first_file:
                with open(output_file_path, "w") as output_file:
                    output_file.write(file_data + "\n")
                first_file = False
            else:
                # 1番目以外のファイルは1行目をスキップして追記
                lines = file_data.split("\n")[1:]
                merged_data = "\n".join(lines)
                with open(output_file_path, "a") as output_file:
                    output_file.write(merged_data + "\n")
    
    input_file = tmp_dir_path + 'input.csv'
    output_file = tmp_dir_path + 'output.csv'

    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        data = list(reader)

    # ラベル列を追加し、行数の分だけ0を入れる
    label_column = ['label']
    for i in range(1, len(data)):
        label_column.append(0)

    # 入力データにラベル列を追加
    for i in range(len(data)):
        data[i].append(label_column[i])

    with open(output_file, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)