####################################################################################################
# SVMモデル作成
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


# 人ごとにSVMモデル作成
ID = 2 #参加者
eer_threshold = {}
outfile = open('Ans_SVM_1.csv','w', newline='')
writer = csv.writer(outfile)
writer.writerow(['ID','AUC', 'EER', '正解率' , '適合率' , '再現率']) 
for i in range(ID):
    ID_n = i+1

    # 学習用のデータのファイルパス
    load_input_data_path = "combined_data_"+str(i+1)+".csv"

    # 学習用のデータを読み込む
    Data = pd.read_csv(load_input_data_path, sep=",")

    # 1列目と2列目のデータを削除する
    Data = Data.drop(Data.columns[[0, 1]], axis=1)

    # 削除後のデータを確認する

    # print(Data.head())

    # 説明変数：x1, x2, x3, x4
    X = Data.loc[:, ['acc_x__median',
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
    'gyro_gamma__number_peaks__n_50'] ].values

    # 目的変数：x5
    Y = Data ['label'].values 

    X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    # X_train=X
    # Y_train=Y
    # X_test=X
    # Y_test=Y

    #標準化
    sc=StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # グリッドサーチによるハイパーパラメータ探索候補設定
    # グリッドサーチのパラメータを設定
    params = [
        {"C": [1,10,100,1000], "kernel":["linear"]},
        {"C": [1,10,100,1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]}
    ]

    # グリッドサーチを行う
    #clf = GridSearchCV( svm.SVC(), params, n_jobs=-1,cv=3,iid=True )
    clf = GridSearchCV( svm.SVC(probability=True,class_weight="balanced"), params, n_jobs=-1,cv=3)
    clf.fit(X_train_std, Y_train)
    # print("学習モデル=", clf.best_estimator_)
    model = clf.best_estimator_
    
    # モデルを保存する
    filename_model = 'binary-SVM-ID{}-rhythm.sav'.format(ID_n)
    pickle.dump(model, open(filename_model, 'wb'))

    # 検証用データで精度を確認
    pred_test = clf.predict(X_test_std)
    accuracy_test = accuracy_score(Y_test, pred_test)
    precision_test = precision_score(Y_test, pred_test)
    TPR = recall_score(Y_test, pred_test)
    #print("正解率=",ac_score)
    tn, fp, fn, tp = confusion_matrix(Y_test, pred_test).ravel() #　混同行列のそれぞれの結果を取得
    print("TN", tn)
    print("FP", fp)
    print("FN", fn)
    print("TP", tp)
    
    #EER算出,ROC描画
    # test_fpr, test_tpr, test_thresholds = roc_curve(Y_test, clf.decision_function(X_test_std))
    test_fpr, test_tpr, test_thresholds = roc_curve(Y_test, model.predict_proba(X_test_std)[:, 1], drop_intermediate=False)
    test_auc = auc(test_fpr, test_tpr)
    test_fnr = 1 - test_tpr
    eer_threshold[ID_n] = test_thresholds[np.nanargmin(np.absolute((test_fnr - test_fpr)))]
    eer = test_fpr[np.nanargmin(np.absolute((test_fnr - test_fpr)))]
    #print("EER:", eer)
    print("eer_threshold:", eer_threshold)
    prob = model.predict_proba(X_test_std)[:, 1]
    
    plt.plot(test_fpr, test_tpr, label="USERID: {}".format(ID_n))
    
    plt.legend()


    writer.writerow([ID_n,test_auc,eer,accuracy_test , precision_test , TPR])

plt.xlabel("FPR",fontsize=20)
plt.ylabel("TPR",fontsize=20)
plt.savefig("svm-roc-toshi.pdf")#pdf化
outfile.close()     

#SVMモデルをdata配下に保存
