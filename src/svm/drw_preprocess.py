import sys
sys.path.append('../')
from myapp.dtw import *
import os


dir_path = "../data/"
tmp_dtw_dir_path = "./tmp_dtw/"

# データを格納するリスト
wave1 = [] # 基準となるデータ
wave2 = [] # それ以外


## 本人同士のDTW算出
with open(dir_path+"user_20230716_211650.csv", "r") as f:
    csv_f = csv.reader(f)
    for i,row in enumerate(csv_f):
        if not i == 0:
            wave1.append(float(row[2]))
with open(dir_path+"user_20230716_213407.csv", "r") as f:
    csv_f = csv.reader(f)
    for i,row in enumerate(csv_f):
        if not i == 0:
            wave2.append(float(row[2]))
nonlinear_distance = nonlinear_dtw(wave1, wave2)
print(nonlinear_distance)
# 出力先のCSVファイル名
output_file = tmp_dtw_dir_path + "combined_data_user.csv"
# ファイルが存在する場合は追記モードで開き、存在しない場合は新規作成して書き込む
mode = 'a' if os.path.exists(output_file) else 'w'
# リストの要素をCSVファイルに書き込む
with open(output_file, mode, newline='') as file:
    writer = csv.writer(file)
    # ファイルが存在しない場合はファイルを作成し、1行だけ書き込む
    if mode == 'w':
        writer.writerow([nonlinear_distance, 1])  # 新しい行を追記
    else:
        writer.writerow([nonlinear_distance, 1])  # 既存のファイルに新しい行を追記



### 他人とのDTW算出
# with open(dir_path+"user_20230716_211650.csv", "r") as f:
#     csv_f = csv.reader(f)
#     for i,row in enumerate(csv_f):
#         if not i == 0:
#             wave1.append(float(row[2]))
# # 出力先CSVファイル名
# output_file = tmp_dtw_dir_path + "combined_data_user_other.csv"
# # ディレクトリ内のファイルを順に処理
# for file_name in os.listdir(dir_path):
#     if "Yuusuke" in file_name:
#         file_path = os.path.join(dir_path, file_name)
#         wave2 = []
#         with open(file_path, "r") as f:
#             csv_f = csv.reader(f)
#             for i, row in enumerate(csv_f):
#                 if not i == 0:
#                     wave2.append(float(row[2]))
#         linear_distance = linear_dtw(wave1, wave2)
#         print(linear_distance)
#         # ファイルが存在する場合は追記モードで開き、存在しない場合は新規作成して書き込む
#         mode = 'a' if os.path.exists(output_file) else 'w'
#         # リストの要素をCSVファイルに書き込む
#         with open(output_file, mode, newline='') as file:
#             writer = csv.writer(file)
#             # ファイルが存在しない場合はファイルを作成し、1行だけ書き込む
#             if mode == 'w':
#                 writer.writerow([linear_distance, 0])  # 新しい行を追記
#             else:
#                 writer.writerow([linear_distance, 0])  # 既存のファイルに新しい行を追記




