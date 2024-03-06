from itertools import count
from tkinter import Label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import sys
sys.path.append('../')
from myapp.dtw import *


dir_path = "../data/"
tmp_dtw_dir_path = "./tmp_dtw/"


def read_csv_column(csv_file, column_index=0):
    data_list = []
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) > column_index:
                data_list.append(row[column_index])
    return data_list

# ERR（FPR = FNRとなる点に一番近い点のthresholdを取り出す）
def compute_eer(fpr,tpr,thresholds):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1-tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer, thresholds[min_index]


dtw_real_user = []
dtw_other_user = []

# ファイル名を指定してリストに保存
csv_file_path_real = tmp_dtw_dir_path + "combined_data_user.csv"
column_index_to_read = 0  # 1列目のデータを読み込む場合は0を指定
dtw_real_user = read_csv_column(csv_file_path_real, column_index_to_read)
csv_file_path_other = tmp_dtw_dir_path + "combined_data_user_other.csv"
dtw_other_user = read_csv_column(csv_file_path_other, column_index_to_read)

dtw_real_50_user = np.array(dtw_real_user).reshape(50,1) # 50行1列
dtw_other_200_user = np.array(dtw_other_user).reshape(203,1) # 203行1列
dtw_250_user = np.concatenate((dtw_other_200_user,dtw_real_50_user),axis=0)

# Label
label_real_50_user = np.ones((50,1),dtype=int)
label_other_200_user = np.zeros((203,1),dtype=int)
label_250_user = np.concatenate((label_other_200_user,label_real_50_user),axis=0)

#Combine Age and Label
data_250_user = np.concatenate((dtw_250_user,label_250_user),axis=1)

#Create row and column
user_name_user = []
for i in range(1,len(dtw_250_user)+1,1):
    user_name_user.append('No.'+str(i))
columns = ['DTW distance','Label']

#Create dataframe
df_data_user = pd.DataFrame(data_250_user,index=user_name_user,columns=columns)
data_user = pd.DataFrame(data_250_user,index=user_name_user,columns=columns)

df_data_user['Label'] = df_data_user['Label'].astype(int)
df_data_user['DTW distance'] = df_data_user['DTW distance'].astype(float)
fpr, tpr, thresholds = roc_curve(df_data_user['Label'],1/df_data_user['DTW distance'])
fig = plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, marker='o')
plt.xlabel('1-Specficity')
plt.ylabel('Sensitivity')
plt.title('user')
plt.show()

print('user')
print('AUC = ', roc_auc_score(df_data_user['Label'],1/df_data_user['DTW distance']))
eer, thresholds = compute_eer(fpr, tpr, thresholds)
print('EER = ', eer)
print('thresholds = ' , thresholds)
thresholds_distance_user = 1/thresholds
print('thresholds(user) = ' , thresholds_distance_user)

wave1=[]
wave2=[]
auth_dir_path = "../data_auth2/"

with open(dir_path+"user_20230716_211650.csv", "r") as f:
    csv_f = csv.reader(f)
    for i,row in enumerate(csv_f):
        if not i == 0:
            wave1.append(float(row[2]))
with open(auth_dir_path+"20230720_081103_user.csv", "r") as f:
    csv_f = csv.reader(f)
    for i,row in enumerate(csv_f):
        if not i == 0:
            wave2.append(float(row[2]))
linear_distance = nonlinear_dtw(wave1, wave2)
print(linear_distance)

