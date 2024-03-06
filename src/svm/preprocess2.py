import numpy as np
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import glob

dir_path = "../data"

# ファイル名に含まれるユーザ名の一覧を取得
user_names = set()
for file_name in os.listdir(dir_path):
    if file_name.endswith(".csv"):
        user_name = file_name.split("_")[0]
        user_names.add(user_name)

# 処理結果を格納する辞書
results = {}

# ユーザごとに処理を行い、結果を辞書に格納
for user_name in user_names:
    # 処理結果を格納するリスト
    user_data = []
    
    # ユーザに関連するファイルを処理
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".csv") and user_name in file_name:
            csv_file = os.path.join(dir_path, file_name)
            
            # CSVファイルをDataFrameとして読み込む
            data_frame = pd.read_csv(csv_file)
            
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
            
            # 処理結果をリストに追加
            user_data.append([rhythm_time, num_peaks_x, num_peaks_y, num_peaks_z, max_peak_x, max_peak_y, max_peak_z, tap_energy])
    
    # ユーザごとの処理結果を辞書に格納
    results[user_name] = user_data

# 結果をファイルに保存
tmp_rhythm_dir_path = "./tmp_rhythm/"
os.makedirs(tmp_rhythm_dir_path, exist_ok=True)
for user_name, user_data in results.items():
    output_file = os.path.join(tmp_rhythm_dir_path, f"rhythm_features_{user_name}.csv")
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["rhythm_time", "num_peaks_x", "num_peaks_y", "num_peaks_z", "max_peak_x", "max_peak_y", "max_peak_z", "tap_energy"])
        writer.writerows(user_data)

tmp_dir_path = "./tmp/"
tmp2_dir_path = "./tmp2/"
# results辞書を順に処理
for user_name, user_data in results.items():
    file1 = f'{tmp_dir_path}extracted_features_{user_name}.csv'
    file2 = f"{tmp_rhythm_dir_path}rhythm_features_{user_name}.csv"

    # CSVファイルをDataFrameとして読み込む
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # ファイル1とファイル2を結合する
    merged_df = pd.concat([df1, df2], axis=1)

    # 結合したDataFrameをCSVファイルとして保存
    merged_df.to_csv(f"{tmp2_dir_path}extracted_features_{user_name}.csv", index=False)


# CSV結合 & ラベル付け
# まとめるファイルのパス
output_file_path = tmp2_dir_path + "input.csv"

# # まとめるファイルのディレクトリパス
# current_directory = os.getcwd()
# directory_path = current_directory

# # まとめるファイルの拡張子
# file_extension = "*.csv"  # 読み込むファイルの拡張子を指定

# ファイルをまとめるための初期化処理
first_file = True

# ディレクトリ内のファイルを順に処理
for file_path in glob.glob(os.path.join(tmp2_dir_path, "*.csv")):
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

input_file = tmp2_dir_path + 'input.csv'
output_file = tmp2_dir_path + 'output.csv'

with open(input_file, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    data = list(reader)

# ラベル列を追加し、行数の分だけ0を入れる
label_column = ['label']
for i in range(1, len(data)):
    label_column.append(0)

# 入力データにラベル列を追加
for i in range(len(data)):
    data[i].extend([label_column[i]])

with open(output_file, 'w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

# 「リズム時間（最後のタップの終了時刻と最初のタップの開始時刻の差）、
# タップエネルギー（3軸のピーク値の2乗を足したもの）、
# タップ回数（ピークが検出された回数）」の3つではなく、
# the tap number, rhythm time, tap interval, and tap energy. の4つ
# 
# 

# dir_path = "../data"

# # 保存先のディレクトリパス
# directory = '../data_svm_model2/'


# # CSVファイルのパス
# csv_file = 'user_20230626_181031.csv'

# # CSVファイルを開く
# with open(dir_path + "/" + csv_file, 'r') as file:
#     reader = csv.reader(file)
#     data = list(reader)


# # 先頭行のTimestampを抜き出す
# first_timestamp = int(data[1][0])  # 先頭行はインデックス1
# # 最終行のTimestampを抜き出す
# last_timestamp = int(data[-1][0])  # 最終行はインデックス-1
# # Timestampの差を計算する
# rhythm_time = last_timestamp - first_timestamp # リズム時間
# # print("先頭行のTimestamp:", first_timestamp)
# # print("最終行のTimestamp:", last_timestamp)
# print("rhythm time:", rhythm_time)


# # CSVファイルをDataFrameとして読み込む
# data_frame = pd.read_csv(dir_path + "/" + csv_file)
# # 0列目のTimestampの値を取得
# timestamps = data_frame.iloc[:, 0]
# # 画面を叩いた場合は3列目のデータを取得（Z軸の加速度データ）
# data_z = data_frame.iloc[:, 3]
# # マイナス方向のピークを検出
# peaks_z, _ = find_peaks(-data_z)
# # マイナス方向のピークのみを抽出
# negative_peaks_z = peaks_z[data_z.iloc[peaks_z] < -2]
# # 主要なピークのみを抽出
# main_peaks_z = []
# peak_timestamps_z = []  # ピークの時刻を格納する配列
# for peak in negative_peaks_z:
#     left = peak - 1
#     right = peak + 1
#     if data_z.iloc[left] < data_z.iloc[peak] or data_z.iloc[right] < data_z.iloc[peak]:
#         continue
#     main_peaks_z.append(peak)
#     peak_timestamps_z.append(timestamps.iloc[peak])  # ピークの時刻を配列に追加
# # グラフの作成とデータのプロット
# plt.plot(data_frame.iloc[:, 0], data_z)
# # マイナス方向のピークに印をつける
# plt.plot(data_frame.iloc[main_peaks_z, 0], data_z.iloc[main_peaks_z], 'ro')
# # ピークの検出回数を表示
# num_peaks_z = len(main_peaks_z) # タップ回数
# print("the tap number Z:",num_peaks_z)
# # マイナス方向のピークの中から最も大きいピークの値を取得
# if len(main_peaks_z) > 0:
#     max_peak_z = data_z.iloc[main_peaks_z].min()
# else:
#     max_peak_z = 0
# # マイナス方向の最も大きいピークの値を表示
# print("Max Peak Z:", max_peak_z)
# plt.text(0.95, 0.95, f"Peaks: {num_peaks_z}", ha='right', va='top', transform=plt.gca().transAxes)


# # 側面を叩いた場合は1列目のデータを取得（X軸の加速度データ）
# data_x = data_frame.iloc[:, 1]
# # マイナス方向のピークを検出
# peaks_x, _ = find_peaks(-data_x)
# # マイナス方向のピークのみを抽出
# negative_peaks_x = peaks_x[data_x.iloc[peaks_x] < -2]
# # 主要なピークのみを抽出
# main_peaks_x = []
# peak_timestamps_x = []  # ピークの時刻を格納する配列
# for peak in negative_peaks_x:
#     left = peak - 1
#     right = peak + 1
#     if data_x.iloc[left] < data_x.iloc[peak] or data_x.iloc[right] < data_x.iloc[peak]:
#         continue
#     main_peaks_x.append(peak)
#     peak_timestamps_x.append(timestamps.iloc[peak])  # ピークの時刻を配列に追加
# # グラフの作成とデータのプロット
# plt.plot(data_frame.iloc[:, 0], data_x)
# # マイナス方向のピークに印をつける
# plt.plot(data_frame.iloc[main_peaks_x, 0], data_x.iloc[main_peaks_x], 'ro')
# # ピークの検出回数を表示
# num_peaks_x = len(main_peaks_x) # タップ回数
# print("the tap number X:",num_peaks_x)
# # マイナス方向のピークの中から最も大きいピークの値を取得
# if len(main_peaks_x) > 0:
#     max_peak_x = data_x.iloc[main_peaks_x].min()
# else:
#     max_peak_x = 0
# # マイナス方向の最も大きいピークの値を表示
# print("Max Peak X:", max_peak_x)
# plt.text(0.95, 0.95, f"Peaks: {num_peaks_x}", ha='right', va='top', transform=plt.gca().transAxes)


# # 2列目のデータを取得（Y軸の加速度データ）
# data_y = data_frame.iloc[:, 1]
# # マイナス方向のピークを検出
# peaks_y, _ = find_peaks(-data_y)
# # マイナス方向のピークのみを抽出
# negative_peaks_y = peaks_y[data_y.iloc[peaks_y] < -2]
# # 主要なピークのみを抽出
# main_peaks_y = []
# peak_timestamps_y = []  # ピークの時刻を格納する配列
# for peak in negative_peaks_y:
#     left = peak - 1
#     right = peak + 1
#     if data_y.iloc[left] < data_y.iloc[peak] or data_y.iloc[right] < data_y.iloc[peak]:
#         continue
#     main_peaks_y.append(peak)
#     peak_timestamps_y.append(timestamps.iloc[peak])  # ピークの時刻を配列に追加
# # グラフの作成とデータのプロット
# plt.plot(data_frame.iloc[:, 0], data_y)
# # マイナス方向のピークに印をつける
# plt.plot(data_frame.iloc[main_peaks_y, 0], data_y.iloc[main_peaks_y], 'ro')
# # ピークの検出回数を表示
# num_peaks_y = len(main_peaks_y) # タップ回数
# print("the tap number Y:",num_peaks_y)
# # マイナス方向のピークの中から最も大きいピークの値を取得
# if len(main_peaks_y) > 0:
#     max_peak_y = data_y.iloc[main_peaks_y].min()
# else:
#     max_peak_y = 0
# # マイナス方向の最も大きいピークの値を表示
# print("Max Peak Y:", max_peak_y)
# plt.text(0.95, 0.95, f"Peaks: {num_peaks_y}", ha='right', va='top', transform=plt.gca().transAxes)

# # タップエネルギー（3軸のピーク値の2乗を足したもの）
# tap_energy = max_peak_x**2 + max_peak_y**2 + max_peak_z**2
# print("tap_energy:", tap_energy)

# ピークの時刻の差を計算して配列に格納
# time_diffs = np.diff(peak_timestamps)

# ピークの最大値、最小値、平均値、中央値を計算
# peak_values = data.iloc[main_peaks]
# max_value = np.max(peak_values)
# min_value = np.min(peak_values)
# mean_value = np.mean(peak_values)
# median_value = np.median(peak_values)

# ピークの時刻の差の配列を表示
# print("Peak Time Differences（タップ間隔）:", time_diffs)
# print("Max Peak Value（最大値）:", max_value)
# print("Min Peak Value（最小値）:", min_value)
# print("Mean Peak Value（平均値）:", mean_value)
# print("Median Peak Value（中央値）:", median_value)

# # グラフの表示
# plt.xlabel('Timestamp')
# plt.ylabel('Data')
# plt.title('Peak Detection - Column 3 (Negative Peaks)')
# plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt

# 時間差を計算
# time_diffs = timestamps.diff().dropna()

# # ヒストグラムの作成
# plt.hist(time_diffs, bins=10)

# # プロットの設定
# plt.xlabel('Time Difference')
# plt.ylabel('Frequency')
# plt.title('Histogram of Time Differences')

# # ヒストグラムの表示
# plt.show()

# # 差の結果を配列に保存
# time_diffs_array = time_diffs.values