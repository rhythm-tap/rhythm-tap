

import torch
print("TORCH VERSION: "+torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchmetrics
import torchsummary
import torchinfo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import os
import re
import sys
import time
import datetime
import random
from pprint import pprint


# シード設定
def setSeed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

setSeed()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU_COUNT = torch.cuda.device_count() if device == 'cuda' else 1
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))


####################################################################################################
# LSTM
####################################################################################################

class LstmFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LstmFeatureExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.ModuleList([nn.LSTM(input_dim if i==0 else hidden_dim, hidden_dim, 1, batch_first=True) for i in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        for i, lstm_layer in enumerate(self.lstm):
            out, (h_n, c_n) = lstm_layer(x, (h0, c0))
            x = self.layer_norms[i](out)
            x = self.dropout(x)
            if i != 0:
                x = x + residual
            residual = x
        out = torch.mean(x, dim=2)
        # out = x.view(x.size(0), -1)
        # out = x[:, -1, :]
        return out

class LstmClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, seq_length):
        super(LstmClassifier, self).__init__()
        self.feature_extractor = LstmFeatureExtractor(input_dim, hidden_dim, num_layers)
        self.linear = nn.Linear(seq_length * 2, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, output_dim)
        self.init_params()

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.uniform_(param.data, -0.08, 0.08)
                    elif 'weight_hh' in name:
                        nn.init.uniform_(param.data, -0.08, 0.08)
                    elif 'bias' in name:
                        param.data.zero_()
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, 0, 0.01)
                module.bias.data.zero_()

    def forward(self, x1, x2):
        out1 = self.feature_extractor(x1)
        out2 = self.feature_extractor(x2)
        out = torch.cat((out1, out2), dim=1)
        out = self.linear(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class PersonalLstmClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, seq_length):
        super(PersonalLstmClassifier, self).__init__()
        self.feature_extractor = LstmFeatureExtractor(input_dim, hidden_dim, num_layers)
        self.linear = nn.Linear(seq_length, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, output_dim)
        self.init_params()

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.uniform_(param.data, -0.08, 0.08)
                    elif 'weight_hh' in name:
                        nn.init.uniform_(param.data, -0.08, 0.08)
                    elif 'bias' in name:
                        param.data.zero_()
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, 0, 0.01)
                module.bias.data.zero_()

    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.linear(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

####################################################################################################









####################################################################################################
# Transformer
####################################################################################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, nhead=4):
        super(TransformerFeatureExtractor, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead),  # You might need to adjust the number of heads
            num_layers=num_layers,
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        out = x[:, -1, :]
        return out

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, nhead=4):
        super(TransformerClassifier, self).__init__()
        self.feature_extractor = TransformerFeatureExtractor(input_dim, hidden_dim, num_layers, nhead)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x1, x2):
        out1 = self.feature_extractor(x1)
        out2 = self.feature_extractor(x2)
        out = torch.cat((out1, out2), dim=1)
        out = self.fc(out)
        return out

class PersonalTransformerClassifier(nn.Module): #未実装
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, nhead=4):
        super(TransformerClassifier, self).__init__()
        self.feature_extractor = TransformerFeatureExtractor(input_dim, hidden_dim, num_layers, nhead)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x1, x2):
        out1 = self.feature_extractor(x1)
        out2 = self.feature_extractor(x2)
        out = torch.cat((out1, out2), dim=1)
        out = self.fc(out)
        return out

####################################################################################################










####################################################################################################
# データ作成
####################################################################################################

# ((csv_path, csv_path), label)のリスト作成
def makeCsvPathList(
        dir_path="../data",
        validation_rate=0,
        test_rate=0.2,
        shuffle=False
    ):

    ##################################################
    # 現在はランダムにtrain,validation,testに分けているが、
    # 本来はもっと細部まで考慮して分割する必要あるかも
    # (タスクによる...)
    ##################################################

    pos_data = []
    neg_data = []
    train_data = []
    validation_data = []
    test_data = []

    # train, validation, testの割合を決定
    train_rate = 1 - validation_rate - test_rate

    # すべてのpathを取得
    csv_path_list = sorted(glob.glob(dir_path+"/*.csv"))
    csv_num = len(csv_path_list)

    # すべての組み合わせのリストを作成
    for i in range(csv_num):
        for j in range(i+1, csv_num):
            csv1_path = os.path.basename(csv_path_list[i])
            csv2_path = os.path.basename(csv_path_list[j])
            csv1_match = re.search(r'^(.*)_\d{8}_\d{6}\.csv$', csv1_path)
            csv2_match = re.search(r'^(.*)_\d{8}_\d{6}\.csv$', csv2_path)
            if (csv1_match is None) or (csv2_match is None):
                continue
            csv1_name = csv1_match[1]
            csv2_name = csv2_match[1]
            if csv1_name==csv2_name:
                label = 1
                pos_data.append([[csv_path_list[i], csv_path_list[j]], label])
            else:
                label = 0
                neg_data.append([[csv_path_list[i], csv_path_list[j]], label])

    # shuffle
    setSeed()
    if shuffle:
        random.shuffle(pos_data)
        random.shuffle(neg_data)

    # train, validation, testに分割()
    s1 = int(len(pos_data)*train_rate)
    s2 = int(len(pos_data)*(train_rate+validation_rate))
    train_data_pos = pos_data[ : s1]
    validation_data_pos = pos_data[s1 : s2]
    test_data_pos = pos_data[s2 : ]
    s1 = int(len(neg_data)*train_rate)
    s2 = int(len(neg_data)*(train_rate+validation_rate))
    train_data_neg = neg_data[ : s1]
    validation_data_neg = neg_data[s1 : s2]
    test_data_neg = neg_data[s2 : ]
    train_data = train_data_pos + train_data_neg
    validation_data = validation_data_pos + validation_data_neg
    test_data = test_data_pos + test_data_neg

    # weight計算
    pos_weight = {}
    pos_weight["train"] = len(train_data_neg) / len(train_data_pos)
    if len(validation_data_pos)==0 or len(validation_data_neg)==0:
        pos_weight["validation"] = 1
    else:
        pos_weight["validation"] = len(validation_data_neg) / len(validation_data_pos)
    if len(test_data_neg)==0 or len(test_data_pos)==0:
        pos_weight["test"] = 1
    else:
        pos_weight["test"] = len(test_data_neg) / len(test_data_pos)

    return (train_data, validation_data, test_data, pos_weight)

class WaveDataset(torch.utils.data.Dataset):
    '''
    CSVファイルパスとラベルの組み合わせのリストからDatasetを作成

    Parameters
    ----------
    data: [[(path,path),label],[(path,path),label],....,[(path,path),label]]
        パス2つとラベルのリスト
    num_classes: int
        分類クラス数

    Returns
    -------
    WaveDatasetインスタンス
    '''

    def __init__(self, data=None, num_classes=2, max_seq_len=3000, use_data=['acc','gyro'], log_flg=False):
        self.data = data
        self.data_num = len(data) if data!=None else 0
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.use_data = use_data
        if 'acc' in use_data:
            self.begin_idx = 1
        else:
            self.begin_idx = 4
        if 'gyro' in use_data:
            self.end_idx = 7
        else:
            self.end_idx = 4
        self.log_flg = log_flg
        # data augmentation
        self.noise_bias = 0.05
        self.scale = 0.3

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        csv_path1, csv_path2 = self.data[idx][0]
        label = self.data[idx][1]

        data1 = self.path2tensor(csv_path1)
        data2 = self.path2tensor(csv_path2)

        out_data = torch.stack([data1, data2],axis=0).float()
        out_label = torch.tensor(label).float()
        if self.num_classes==2:
            out_label = out_label.view(-1)
        else:
            out_label = F.one_hot(out_label,num_classes=self.num_classes)

        return out_data, out_label

    def path2tensor(self, csv_path):
        csv_data = pd.read_csv(csv_path, header=None).apply(pd.to_numeric, errors='coerce').values
        data = csv_data[:,self.begin_idx:self.end_idx]
        # gyroデータの正規化
        if 'gyro' in self.use_data:
            data_gyro_normalized = self.normalize_gyro(data[:,-3:])
            data = np.concatenate([data[:,:-3], data_gyro_normalized], axis=1)
        # Padding
        if data.shape[0] < self.max_seq_len:
            data = np.pad(data, ((0,self.max_seq_len-data.shape[0]),(0,0)), 'constant')
        else:
            if self.log_flg:
                print_log(f"Data length exceeds the MAX_SEQ_LEN, trimming from {data.shape[0]} to {self.max_seq_len} ({csv_path}).")
            data = data[:self.max_seq_len, :]
        return torch.from_numpy(data)

    def normalize_gyro(self, data_gyro):
        MAX_ALPHA = 360
        MIN_ALPHA = 0
        MAX_BETA = 180
        MIN_BETA = -180
        MAX_GAMMA = 90
        MIN_GAMMA = -90
        alpha = data_gyro[:, 0]
        beta = data_gyro[:, 1]
        gamma = data_gyro[:, 2]
        # alphaの正規化（0～1の範囲にスケーリング）
        alpha_normalized = (alpha - MIN_ALPHA) / (MAX_ALPHA - MIN_ALPHA)
        # betaの正規化（-1～1の範囲にスケーリング）
        beta_normalized = (beta - MIN_BETA) / (MAX_BETA - MIN_BETA) * 2 - 1
        # gammaの正規化（-1～1の範囲にスケーリング）
        gamma_normalized = (gamma - MIN_GAMMA) / (MAX_GAMMA - MIN_GAMMA) * 2 - 1
        data_gyro_normalized = np.stack((alpha_normalized, beta_normalized, gamma_normalized), axis=1)
        return data_gyro_normalized

    def apply_augmentation(self, data):
        # ノイズの追加
        noise = np.random.randn(*data.shape) * self.noise_bias
        data = data + noise
        # データのスケーリング(1-self.scale〜1+self.scaleの範囲でスケーリング)
        scale = np.random.rand() * self.scale*2 + (1-self.scale)
        data = data * scale
        return data


# (csv_path, label)のリスト作成
def makePersonalCsvPathList(
        personal_name,
        dir_path="../data",
        validation_rate=0,
        test_rate=0.2,
        shuffle=False
    ):

    ##################################################
    # 現在はランダムにtrain,validation,testに分けているが、
    # 本来はもっと細部まで考慮して分割する必要あるかも
    # (タスクによる...)
    ##################################################

    pos_data = []
    neg_data = []
    train_data = []
    validation_data = []
    test_data = []

    # train, validation, testの割合を決定
    train_rate = 1 - validation_rate - test_rate

    # すべてのpathを取得
    csv_path_list = sorted(glob.glob(dir_path+"/*.csv"))
    csv_num = len(csv_path_list)

    # CSVとlabelのリストを作成
    for i in range(csv_num):
        csv_path = os.path.basename(csv_path_list[i])
        csv_match = re.search(r'^(.*)_\d{8}_\d{6}\.csv$', csv_path)
        if csv_match is None:
            continue
        csv_name = csv_match[1]
        if csv_name==personal_name:
            label = 1
            pos_data.append([csv_path_list[i], label])
        else:
            label = 0
            neg_data.append([csv_path_list[i], label])

    # shuffle
    setSeed()
    if shuffle:
        random.shuffle(pos_data)
        random.shuffle(neg_data)

    # train, validation, testに分割()
    s1 = int(len(pos_data)*train_rate)
    s2 = int(len(pos_data)*(train_rate+validation_rate))
    train_data_pos = pos_data[ : s1]
    validation_data_pos = pos_data[s1 : s2]
    test_data_pos = pos_data[s2 : ]
    s1 = int(len(neg_data)*train_rate)
    s2 = int(len(neg_data)*(train_rate+validation_rate))
    train_data_neg = neg_data[ : s1]
    validation_data_neg = neg_data[s1 : s2]
    test_data_neg = neg_data[s2 : ]
    train_data = train_data_pos + train_data_neg
    validation_data = validation_data_pos + validation_data_neg
    test_data = test_data_pos + test_data_neg

    # weight計算
    pos_weight = {}
    pos_weight["train"] = len(train_data_neg) / len(train_data_pos)
    if len(validation_data_pos)==0 or len(validation_data_neg)==0:
        pos_weight["validation"] = 1
    else:
        pos_weight["validation"] = len(validation_data_neg) / len(validation_data_pos)
    if len(test_data_neg)==0 or len(test_data_pos)==0:
        pos_weight["test"] = 1
    else:
        pos_weight["test"] = len(test_data_neg) / len(test_data_pos)

    return (train_data, validation_data, test_data, pos_weight)

def makeFilteredPersonalCsvPathList(
        personal_name,
        names_list,
        dir_path_1="../data/dir1",
        dir_path_2="../data/dir2",
        validation_rate=0,
        test_rate=0.2,
        shuffle=False
    ):

    pos_data = []
    neg_data = []

    # train, validation, testの割合を決定
    train_rate = 1 - validation_rate - test_rate
    csv_path_list_1 = sorted(glob.glob(dir_path_1 + "/*.csv"))
    csv_path_list_2 = sorted(glob.glob(dir_path_2 + "/*.csv"))

    # 2つのリストを結合
    csv_path_list = csv_path_list_1 + csv_path_list_2

    # CSVとlabelのリストを作成（名前リストに基づくフィルタリング）
    for csv_path_full in csv_path_list:
        csv_path = os.path.basename(csv_path_full)
        csv_match = re.search(r'^(.*)_\d{8}_\d{6}\.csv$', csv_path)
        if csv_match is None:
            continue
        csv_name = csv_match[1]
        if csv_name in names_list:
            label = 1 if csv_name == personal_name else 0
            if label == 1:
                pos_data.append([csv_path_full, label])
            else:
                neg_data.append([csv_path_full, label])

    # shuffle
    setSeed()
    if shuffle:
        random.shuffle(pos_data)
        random.shuffle(neg_data)

    # train, validation, testに分割()
    s1 = int(len(pos_data)*train_rate)
    s2 = int(len(pos_data)*(train_rate+validation_rate))
    train_data_pos = pos_data[ : s1]
    validation_data_pos = pos_data[s1 : s2]
    test_data_pos = pos_data[s2 : ]
    s1 = int(len(neg_data)*train_rate)
    s2 = int(len(neg_data)*(train_rate+validation_rate))
    train_data_neg = neg_data[ : s1]
    validation_data_neg = neg_data[s1 : s2]
    test_data_neg = neg_data[s2 : ]
    train_data = train_data_pos + train_data_neg
    validation_data = validation_data_pos + validation_data_neg
    test_data = test_data_pos + test_data_neg

    # weight計算
    pos_weight = {}
    pos_weight["train"] = len(train_data_neg) / len(train_data_pos)
    if len(validation_data_pos)==0 or len(validation_data_neg)==0:
        pos_weight["validation"] = 1
    else:
        pos_weight["validation"] = len(validation_data_neg) / len(validation_data_pos)
    if len(test_data_neg)==0 or len(test_data_pos)==0:
        pos_weight["test"] = 1
    else:
        pos_weight["test"] = len(test_data_neg) / len(test_data_pos)

    return (train_data, validation_data, test_data, pos_weight)


class PersonalWaveDataset(torch.utils.data.Dataset):
    '''
    CSVファイルパスとラベルの組み合わせのリストからDatasetを作成

    Parameters
    ----------
    data: [[path,label],[path,label],....,[path,label]]
        パス1つとラベルのリスト
    num_classes: int
        分類クラス数

    Returns
    -------
    PersonalWaveDatasetインスタンス
    '''

    def __init__(self, data=None, num_classes=2, max_seq_len=3000, use_data=['acc','gyro'], log_flg=False):
        self.data = data
        self.data_num = len(data) if data!=None else 0
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.use_data = use_data
        if 'acc' in use_data:
            self.begin_idx = 1
        else:
            self.begin_idx = 4
        if 'gyro' in use_data:
            self.end_idx = 7
        else:
            self.end_idx = 4
        self.log_flg = log_flg
        # data augmentation
        self.noise_bias = 0.05
        self.scale = 0.3

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        csv_path = self.data[idx][0]
        label = self.data[idx][1]

        data = self.path2tensor(csv_path)

        out_data = data.float()
        out_label = torch.tensor(label).float()
        if self.num_classes==2:
            out_label = out_label.view(-1)
        else:
            out_label = F.one_hot(out_label,num_classes=self.num_classes)

        return out_data, out_label

    def path2tensor(self, csv_path):
        csv_data = pd.read_csv(csv_path, header=None).apply(pd.to_numeric, errors='coerce').values
        data = csv_data[:,self.begin_idx:self.end_idx]
        data = self.apply_augmentation(data)
        # gyroデータの正規化
        if 'gyro' in self.use_data:
            data_gyro_normalized = self.normalize_gyro(data[:,-3:])
            data = np.concatenate([data[:,:-3], data_gyro_normalized], axis=1)
        # Padding
        if data.shape[0] < self.max_seq_len:
            data = np.pad(data, ((0,self.max_seq_len-data.shape[0]),(0,0)), 'constant')
        else:
            if self.log_flg:
                print_log(f"Data length exceeds the MAX_SEQ_LEN, trimming from {data.shape[0]} to {self.max_seq_len} ({csv_path}).")
            data = data[:self.max_seq_len, :]
        return torch.from_numpy(data)

    def normalize_gyro(self, data_gyro):
        MAX_ALPHA = 360
        MIN_ALPHA = 0
        MAX_BETA = 180
        MIN_BETA = -180
        MAX_GAMMA = 90
        MIN_GAMMA = -90
        alpha = data_gyro[:, 0]
        beta = data_gyro[:, 1]
        gamma = data_gyro[:, 2]
        # alphaの正規化（0～1の範囲にスケーリング）
        alpha_normalized = (alpha - MIN_ALPHA) / (MAX_ALPHA - MIN_ALPHA)
        # betaの正規化（-1～1の範囲にスケーリング）
        beta_normalized = (beta - MIN_BETA) / (MAX_BETA - MIN_BETA) * 2 - 1
        # gammaの正規化（-1～1の範囲にスケーリング）
        gamma_normalized = (gamma - MIN_GAMMA) / (MAX_GAMMA - MIN_GAMMA) * 2 - 1
        data_gyro_normalized = np.stack((alpha_normalized, beta_normalized, gamma_normalized), axis=1)
        return data_gyro_normalized

    def apply_augmentation(self, data):
        # ノイズの追加
        noise = np.random.randn(*data.shape) * self.noise_bias
        data = data + noise
        # データのスケーリング(1-self.scale〜1+self.scaleの範囲でスケーリング)
        scale = np.random.rand() * self.scale*2 + (1-self.scale)
        data = data * scale
        return data

####################################################################################################






####################################################################################################
# 便利関数
####################################################################################################

### ログ表示 ###
def print_log(message="", line_break=True):
    if line_break:
        sys.stdout.write(message + "\n")
    else:
        sys.stdout.write(message)
    sys.stdout.flush()


### 学習(1epoch) ###
def train_one_epoch(model, device, train_dataloader, loss_fn, optimizer, callbacks=None, metrics_dict={}):
    """
    pytorchのモデルを1エポックだけ学習する。
    
    Parameters
    ----------
    model : torch.nn.Module
        学習対象のモデル
    device : torch.cuda.device or str
        使用するデバイス
    train_dataloder : torch.utils.data.DataLoader
        学習データ
    loss_fn : torch.nn.lossFunctions
        損失関数
    optimizer : torch.optim.*
        最適化関数
    callback : dict or None
        コールバック関数。実行したい位置によって辞書のkeyを指定する。
            on_train_begin      -> 学習開始前
            on_train_end        -> 学習終了後
    metrics_dict : dict
        計算したいMetricsの辞書を指定。getMetrics関数によって作成されたものをそのまま指定。

    Returns
    -------
    history_epoch : dict
        学習結果
    """

    # 変数
    bar_length = 30 # プログレスバーの長さ
    train_batch_num = len(train_dataloader) # batchの総数

    # metricsからROCは削除
    if 'roc' in metrics_dict:
        metrics_dict.pop('roc')

    # metrics
    metrics = {}
    for k, _ in metrics_dict.items():
        metrics[k] = []

    # callbackの実行
    if (callbacks is not None) and ('on_train_begin' in callbacks) and callable(callbacks['on_train_begin']):
        callbacks['on_train_begin'](model=model)

    # モデルを訓練モードにする
    model.train()

    history_epoch = {}
    losses = []
    t_train = 0

    batch_start_time = time.time()
    for batch_idx, (inputs, labels) in enumerate(train_dataloader):

        metric_batch = {}
        inputs1, inputs2 = inputs[:,0,:,:], inputs[:,1,:,:]
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

        # optimizerを初期化
        optimizer.zero_grad()

        # ニューラルネットワークの処理を行う
        outputs = model(inputs1, inputs2)

        # 損失(出力とラベルとの誤差)の計算
        loss = loss_fn(outputs, labels)
        losses.append(loss.item())
        metric_batch['loss'] = loss.item()

        # その他のMetricsの計算
        for k, fn in metrics_dict.items():
            metric = fn(outputs,labels)
            metrics[k].append(metric.item())
            metric_batch[k] = metric.item()

        # 勾配の計算
        loss.backward()
        # for name, param in model.named_parameters():
        #     print(name, param.grad)

        # 重みの更新
        optimizer.step()

        # プログレスバー表示
        interval = time.time() - batch_start_time
        t_train += interval
        eta = str(datetime.timedelta(seconds= int((train_batch_num-batch_idx+1)*interval) ))
        done = math.floor(bar_length * batch_idx / train_batch_num)
        bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
        print_log(f"\r  \033[K[{bar}] - ETA: {eta:>8}, {math.floor(batch_idx / train_batch_num*100):>3}% ({batch_idx}/{train_batch_num})", line_break=False)

        # 学習状況の表示
        for k,v in metric_batch.items():
            print_log(f", {k.capitalize()}: {v:.04f}", line_break=False)

        batch_start_time = time.time()

    done = bar_length
    bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
    print_log(f"\r  \033[K[{bar}] - {int(t_train)}s, 100% ({train_batch_num}/{train_batch_num})", line_break=False)

    # 学習状況の表示&保存
    history_epoch['training_elapsed_time'] = t_train
    train_loss = sum(losses) / len(losses)
    print_log(f", Loss: {train_loss:.04f}", line_break=False)
    history_epoch['loss'] = train_loss
    for k,v in metrics.items():
        train_metric = sum(v) / len(v)
        exec('{} = {}'.format(k, train_metric))
        print_log(f", {k.capitalize()}: {train_metric:.04f}", line_break=False)
        history_epoch[k] = train_metric
    print_log()

    # callbackの実行
    if (callbacks is not None) and ('on_train_end' in callbacks) and callable(callbacks['on_train_end']):
        callbacks['on_train_end'](model=model)

    return history_epoch


### 検証(1epoch) ###
def validation_one_epoch(model, device, validation_dataloader, loss_fn, callbacks=None, metrics_dict={}):
    """
    pytorchのモデルを1エポックだけ検証する。
    
    Parameters
    ----------
    model : torch.nn.Module
        学習対象のモデル
    device : torch.cuda.device or str
        使用するデバイス
    validation_dataloder : torch.utils.data.DataLoader
        検証データ
    loss_fn : torch.nn.lossFunctions
        損失関数
    callback : dict or None
        コールバック関数。実行したい位置によって辞書のkeyを指定する。
            on_validation_begin -> 検証開始前
            on_validation_end   -> 検証終了後
    metrics_dict : dict
        計算したいMetricsの辞書を指定。getMetrics関数によって作成されたものをそのまま指定。

    Returns
    -------
    history_epoch : dict
        検証結果
    """

    # 変数
    bar_length = 30 # プログレスバーの長さ
    val_batch_num = len(validation_dataloader) # batchの総数

    # metricsからROCは削除
    if 'roc' in metrics_dict:
        metrics_dict.pop('roc')

    # metrics
    val_metrics = {}
    for k, _ in metrics_dict.items():
        val_metrics[k] = []

    # callbackの実行
    if (callbacks is not None) and ('on_validation_begin' in callbacks) and callable(callbacks['on_validation_begin']):
        callbacks['on_validation_begin'](model=model)

    # モデルを評価モードにする
    model.eval()

    history_epoch = {}
    val_losses = []
    t_validation = 0

    with torch.no_grad():

        batch_start_time = time.time()
        for batch_idx, (inputs, labels) in enumerate(validation_dataloader):

            inputs1, inputs2 = inputs[:,0,:,:], inputs[:,1,:,:]
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

            # ニューラルネットワークの処理を行う
            outputs = model(inputs1, inputs2)

            # 損失(出力とラベルとの誤差)の計算
            val_loss = loss_fn(outputs, labels)
            val_losses.append(val_loss.item())

            # その他のMetricsの計算
            for k, fn in metrics_dict.items():
                val_metric = fn(outputs,labels)
                val_metrics[k].append(val_metric.item())

            # プログレスバー表示
            interval = time.time() - batch_start_time
            t_validation += interval
            eta = str(datetime.timedelta(seconds= int((val_batch_num-batch_idx+1)*interval) ))
            done = math.floor(bar_length * batch_idx / val_batch_num)
            bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
            print_log(f"\r  \033[K[{bar}] - ETA: {eta:>8}, {math.floor(batch_idx / val_batch_num*100):>3}% ({batch_idx}/{val_batch_num})", line_break=False)

            batch_start_time = time.time()

        done = bar_length
        bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
        print_log(f"\r  \033[K[{bar}] - {int(t_validation)}s, 100% ({val_batch_num}/{val_batch_num})", line_break=False)

        # 検証の表示&保存
        history_epoch['val_elapsed_time'] = t_validation
        val_loss = sum(val_losses) / len(val_losses)
        print_log(f", ValLoss: {val_loss:.04f}", line_break=False)
        history_epoch['val_loss'] = val_loss
        for k,v in val_metrics.items():
            val_metric = sum(v) / len(v)
            exec('{} = {}'.format("val_"+k, val_metric))
            print_log(f", Val{k.capitalize()}: {val_metric:.04f}", line_break=False)
            history_epoch["val_"+k] = val_metric
        print_log()

    # callbackの実行
    if (callbacks is not None) and ('on_validation_end' in callbacks) and callable(callbacks['on_validation_end']):
        callbacks['on_validation_end'](model=model)

    return history_epoch


### 学習(1epoch) ###
def train_one_epoch_personal(model, device, train_dataloader, loss_fn, optimizer, callbacks=None, metrics_dict={}):
    """
    pytorchのモデルを1エポックだけ学習する。
    
    Parameters
    ----------
    model : torch.nn.Module
        学習対象のモデル
    device : torch.cuda.device or str
        使用するデバイス
    train_dataloder : torch.utils.data.DataLoader
        学習データ
    loss_fn : torch.nn.lossFunctions
        損失関数
    optimizer : torch.optim.*
        最適化関数
    callback : dict or None
        コールバック関数。実行したい位置によって辞書のkeyを指定する。
            on_train_begin      -> 学習開始前
            on_train_end        -> 学習終了後
    metrics_dict : dict
        計算したいMetricsの辞書を指定。getMetrics関数によって作成されたものをそのまま指定。

    Returns
    -------
    history_epoch : dict
        学習結果
    """

    # 変数
    bar_length = 30 # プログレスバーの長さ
    train_batch_num = len(train_dataloader) # batchの総数

    # metricsからROCは削除
    if 'roc' in metrics_dict:
        metrics_dict.pop('roc')

    # metrics
    metrics = {}
    for k, _ in metrics_dict.items():
        metrics[k] = []

    # callbackの実行
    if (callbacks is not None) and ('on_train_begin' in callbacks) and callable(callbacks['on_train_begin']):
        callbacks['on_train_begin'](model=model)

    # モデルを訓練モードにする
    model.train()

    history_epoch = {}
    losses = []
    t_train = 0

    batch_start_time = time.time()
    for batch_idx, (inputs, labels) in enumerate(train_dataloader):

        metric_batch = {}
        inputs, labels = inputs.to(device), labels.to(device)

        # optimizerを初期化
        optimizer.zero_grad()

        # ニューラルネットワークの処理を行う
        outputs = model(inputs)

        # 損失(出力とラベルとの誤差)の計算
        loss = loss_fn(outputs, labels)
        losses.append(loss.item())
        metric_batch['loss'] = loss.item()

        # その他のMetricsの計算
        for k, fn in metrics_dict.items():
            metric = fn(outputs,labels)
            metrics[k].append(metric.item())
            metric_batch[k] = metric.item()

        # 勾配の計算
        loss.backward()
        # for name, param in model.named_parameters():
        #     print(name, param.grad)

        # 重みの更新
        optimizer.step()

        # プログレスバー表示
        interval = time.time() - batch_start_time
        t_train += interval
        eta = str(datetime.timedelta(seconds= int((train_batch_num-batch_idx+1)*interval) ))
        done = math.floor(bar_length * batch_idx / train_batch_num)
        bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
        print_log(f"\r  \033[K[{bar}] - ETA: {eta:>8}, {math.floor(batch_idx / train_batch_num*100):>3}% ({batch_idx}/{train_batch_num})", line_break=False)

        # 学習状況の表示
        for k,v in metric_batch.items():
            print_log(f", {k.capitalize()}: {v:.04f}", line_break=False)

        batch_start_time = time.time()

    done = bar_length
    bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
    print_log(f"\r  \033[K[{bar}] - {int(t_train)}s, 100% ({train_batch_num}/{train_batch_num})", line_break=False)

    # 学習状況の表示&保存
    history_epoch['training_elapsed_time'] = t_train
    train_loss = sum(losses) / len(losses)
    print_log(f", Loss: {train_loss:.04f}", line_break=False)
    history_epoch['loss'] = train_loss
    for k,v in metrics.items():
        train_metric = sum(v) / len(v)
        exec('{} = {}'.format(k, train_metric))
        print_log(f", {k.capitalize()}: {train_metric:.04f}", line_break=False)
        history_epoch[k] = train_metric
    print_log()

    # callbackの実行
    if (callbacks is not None) and ('on_train_end' in callbacks) and callable(callbacks['on_train_end']):
        callbacks['on_train_end'](model=model)

    return history_epoch


### 検証(1epoch) ###
def validation_one_epoch_personal(model, device, validation_dataloader, loss_fn, callbacks=None, metrics_dict={}):
    """
    pytorchのモデルを1エポックだけ検証する。
    
    Parameters
    ----------
    model : torch.nn.Module
        学習対象のモデル
    device : torch.cuda.device or str
        使用するデバイス
    validation_dataloder : torch.utils.data.DataLoader
        検証データ
    loss_fn : torch.nn.lossFunctions
        損失関数
    callback : dict or None
        コールバック関数。実行したい位置によって辞書のkeyを指定する。
            on_validation_begin -> 検証開始前
            on_validation_end   -> 検証終了後
    metrics_dict : dict
        計算したいMetricsの辞書を指定。getMetrics関数によって作成されたものをそのまま指定。

    Returns
    -------
    history_epoch : dict
        検証結果
    """

    # 変数
    bar_length = 30 # プログレスバーの長さ
    val_batch_num = len(validation_dataloader) # batchの総数

    # metricsからROCは削除
    if 'roc' in metrics_dict:
        metrics_dict.pop('roc')

    # metrics
    val_metrics = {}
    for k, _ in metrics_dict.items():
        val_metrics[k] = []

    # callbackの実行
    if (callbacks is not None) and ('on_validation_begin' in callbacks) and callable(callbacks['on_validation_begin']):
        callbacks['on_validation_begin'](model=model)

    # モデルを評価モードにする
    model.eval()

    history_epoch = {}
    val_losses = []
    t_validation = 0

    with torch.no_grad():

        batch_start_time = time.time()
        for batch_idx, (inputs, labels) in enumerate(validation_dataloader):

            inputs, labels = inputs.to(device), labels.to(device)

            # ニューラルネットワークの処理を行う
            outputs = model(inputs)

            # 損失(出力とラベルとの誤差)の計算
            val_loss = loss_fn(outputs, labels)
            val_losses.append(val_loss.item())

            # その他のMetricsの計算
            for k, fn in metrics_dict.items():
                val_metric = fn(outputs,labels)
                val_metrics[k].append(val_metric.item())

            # プログレスバー表示
            interval = time.time() - batch_start_time
            t_validation += interval
            eta = str(datetime.timedelta(seconds= int((val_batch_num-batch_idx+1)*interval) ))
            done = math.floor(bar_length * batch_idx / val_batch_num)
            bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
            print_log(f"\r  \033[K[{bar}] - ETA: {eta:>8}, {math.floor(batch_idx / val_batch_num*100):>3}% ({batch_idx}/{val_batch_num})", line_break=False)

            batch_start_time = time.time()

        done = bar_length
        bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
        print_log(f"\r  \033[K[{bar}] - {int(t_validation)}s, 100% ({val_batch_num}/{val_batch_num})", line_break=False)

        # 検証の表示&保存
        history_epoch['val_elapsed_time'] = t_validation
        val_loss = sum(val_losses) / len(val_losses)
        print_log(f", ValLoss: {val_loss:.04f}", line_break=False)
        history_epoch['val_loss'] = val_loss
        for k,v in val_metrics.items():
            val_metric = sum(v) / len(v)
            exec('{} = {}'.format("val_"+k, val_metric))
            print_log(f", Val{k.capitalize()}: {val_metric:.04f}", line_break=False)
            history_epoch["val_"+k] = val_metric
        print_log()

    # callbackの実行
    if (callbacks is not None) and ('on_validation_end' in callbacks) and callable(callbacks['on_validation_end']):
        callbacks['on_validation_end'](model=model)

    return history_epoch


### 評価 ###
def test(model, device, test_dataloader, loss_fn, callbacks=None, metrics_dict={}):
    """
    pytorchのモデルをテストする。
    
    Parameters
    ----------
    model : torch.nn.Module
        テスト対象のモデル
    device : torch.cuda.device or str
        使用するデバイス
    test_dataloder : torch.utils.data.DataLoader
        テストデータ
    loss_fn : torch.nn.lossFunctions
        損失関数
    callback : dict or None
        コールバック関数。実行したい位置によって辞書のkeyを指定する。
            on_test_begin  -> エポック開始前
            on_test_end    -> エポック終了後
    metrics_dict : dict
        計算したいMetricsの辞書を指定。getMetrics関数によって作成されたものをそのまま指定。

    Returns
    -------
    history : dict
        テスト結果
    """

    # 変数
    bar_length = 30 # プログレスバーの長さ
    batch_num = len(test_dataloader) # batchの総数
    history = {} # 返り値用変数

    # モデルを評価モードにする
    model.eval()

    # metrics初期化用
    metrics = {}
    for k, _ in metrics_dict.items():
        if k!='roc':
            metrics[k] = []

    losses = []
    t_test = 0

    # callbackの実行
    if (callbacks is not None) and ('on_test_begin' in callbacks) and callable(callbacks['on_test_begin']):
        callbacks['on_test_begin']()

    print_log(f"Test:")
    with torch.no_grad():

        batch_start_time = time.time()
        for batch_idx, (inputs, labels) in enumerate(test_dataloader):

            inputs, labels = inputs.to(device), labels.to(device)

            # ニューラルネットワークの処理を行う
            outputs = model(inputs)

            # 損失(出力とラベルとの誤差)の計算
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())

            # その他のMetricsの計算
            for k, fn in metrics_dict.items():
                if k=='roc':
                    fn.update(outputs,labels.long())
                else:
                    metric = fn(outputs,labels)
                    metrics[k].append(metric.item())

            # プログレスバー表示
            interval = time.time() - batch_start_time
            t_test += interval
            eta = str(datetime.timedelta(seconds= int((batch_num-batch_idx+1)*interval) ))
            done = math.floor(bar_length * batch_idx / batch_num)
            bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
            print_log(f"\r  \033[K[{bar}] - ETA: {eta:>8}, {math.floor(batch_idx / batch_num*100):>3}% ({batch_idx}/{batch_num})", line_break=False)

            batch_start_time = time.time()

        done = bar_length
        bar = '='*done + ("=" if done == bar_length else ">")  + "."*(bar_length-done)
        print_log(f"\r  \033[K[{bar}] - {int(t_test)}s, 100% ({batch_num}/{batch_num})", line_break=False)

        # 学習状況の表示&保存
        history['test_elapsed_time'] = t_test
        test_loss = sum(losses) / len(losses)
        print_log(f", Loss: {test_loss:.04f}", line_break=False)
        history['loss'] = test_loss
        for k,v in metrics.items():
            if k!='roc':
                test_metric = sum(v) / len(v)
                print_log(f", {k.capitalize()}: {test_metric:.04f}", line_break=False)
                history[k] = test_metric
        if 'roc' in metrics_dict:
            history['roc'] = metrics_dict['roc'].compute()
        print_log()

    # callbackの実行
    if (callbacks is not None) and ('on_test_end' in callbacks) and callable(callbacks['on_test_end']):
        callbacks['on_test_end']()

    return history


### Metricsリスト取得 ###
def getMetrics(device, mode='all', num_classes=2, average='micro'):
    """
    Metricsの辞書リストを取得する
    
    Parameters
    ----------
    mode : str or list
        取得したいMetricsの文字列、またはそのリストを指定。
            'all'
            'accuracy'
    num_classes : int
        ラベル数の指定
    average : str or None
        平均化のタイプを指定。以下tensorflowによる説明
            None        -> 平均化は実行されず、各クラスのスコアが返される。
            'micro'     -> TP,FP,TN,FNなどの合計を数えることによって、グローバルに計算する。
            'macro'     -> 各ラベルのメトリックを計算し、重み付けされていない平均を返す。これはラベルの不均衡を考慮していない。
            'weighted'  -> 各ラベルのメトリックを計算し、各ラベルの真のインスタンス数によって重み付けされた平均を返す。

    Returns
    -------
    metrics_dict : dict
        Metricsの辞書リスト
    """

    task = "binary" if num_classes==2 else "multilabel"
    metrics_dict = {}
    if mode=='all' or ('all' in mode) or mode=='accuracy' or ('accuracy' in mode):
        metrics_dict['accuracy'] = torchmetrics.Accuracy(task=task,num_labels=num_classes).to(device)
    if mode=='all' or ('all' in mode) or mode=='auc' or ('auc' in mode):
        metrics_dict['auc'] = torchmetrics.AUROC(task=task,num_labels=num_classes).to(device)
    if mode=='all' or ('all' in mode) or mode=='precision' or ('precision' in mode):
        metrics_dict['precision'] = torchmetrics.Precision(task=task,num_labels=num_classes).to(device)
    if mode=='all' or ('all' in mode) or mode=='recall' or ('recall' in mode):
        metrics_dict['recall'] = torchmetrics.Recall(task=task,num_labels=num_classes).to(device)
    if mode=='all' or ('all' in mode) or mode=='specificity' or ('specificity' in mode):
        metrics_dict['specificity'] = torchmetrics.Specificity(task=task,num_labels=num_classes).to(device)
    if mode=='all' or ('all' in mode) or mode=='f1' or ('f1' in mode):
        metrics_dict['f1'] = torchmetrics.F1Score(task=task,num_labels=num_classes).to(device)
    if mode=='all' or ('all' in mode) or mode=='roc' or ('roc' in mode):
        metrics_dict['roc'] = torchmetrics.ROC(task=task,num_labels=num_classes).to(device)
    return metrics_dict


### 結果描画 ###
def saveLossGraph(graph_data, save_path='result.png', title='Model accuracy'):
    plt.clf()
    mpl_color_list = ["blue","orange","green","red","purple","brown","pink","gray","olive","cyan"]
    for i, (metrics_name, data) in enumerate(graph_data.items()):
        plt.plot(range(1, len(data)+1), data, color=mpl_color_list[i], linestyle='solid', marker='o', label=metrics_name)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.clf()


### ROC曲線描画 ###
def saveRocCurve(roc_data, save_path='roc.png', title='ROC Curve', fontsize=14):
    plt.figure(figsize=(6.8,4.8))
    plt.clf()
    mpl_color_list = ["blue","orange","green","red","purple","brown","pink","gray","olive","cyan"]
    fpr, tpr, _thresholds = roc_data
    if type(fpr) == list:
        for i in range(len(fpr)):
            fpr[i] = fpr[i].cpu().numpy()
            tpr[i] = tpr[i].cpu().numpy()
            plt.plot(fpr[i], tpr[i], color=mpl_color_list[i], linestyle='solid', marker='', label=f'label {i}')
    else:
        fpr = fpr.cpu().numpy()
        tpr = tpr.cpu().numpy()
        plt.plot(fpr, tpr, color=mpl_color_list[0], linestyle='solid', marker='')
    plt.title(title, fontsize=fontsize+2)
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.savefig(save_path)
    plt.clf()

# def saveRocCurves(roc_data_dict, save_path='rocs.png', title='ROC Curves', chance_level=False, mean=False, std=False):
#     plt.clf()
#     mpl_color_list = [
#         "blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan",
#         "navy", "yellow", "lime", "maroon", "magenta", "teal", "coral", "darkorange", "darkblue", "darkgreen"
#     ]
#     for i, (roc_name, roc_data) in enumerate(roc_data_dict.items()):
#         fpr, tpr, _thresholds = roc_data
#         if hasattr(fpr, 'cpu'):
#             fpr = fpr.cpu().numpy()
#         if hasattr(tpr, 'cpu'):
#             tpr = tpr.cpu().numpy()
#         color_idx = i % len(mpl_color_list)
#         plt.plot(fpr, tpr, color=mpl_color_list[color_idx], linestyle='solid', marker='', label=f'User{i+1}')
#     plt.title(title)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.legend()
#     plt.savefig(save_path)
#     plt.clf()

def saveRocCurves(roc_data_dict, save_path='rocs.png', title='ROC Curves', chance_level=False, mean=False, std=False, auc=False, fontsize=14):
    plt.figure(figsize=(6.8,4.8))
    plt.clf()
    mpl_color_list = [
        "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan", "navy", 
        "yellow", "lime", "maroon", "magenta", "teal", "coral", "darkorange", "darkblue", "darkgreen", "skyblue"
    ]

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for i, (roc_name, roc_data) in enumerate(roc_data_dict.items()):
        fpr, tpr, _thresholds = roc_data
        if hasattr(fpr, 'cpu'):
            fpr = fpr.cpu().numpy()
        if hasattr(tpr, 'cpu'):
            tpr = tpr.cpu().numpy()

        color_idx = i % len(mpl_color_list)
        if auc:
            plt.plot(fpr, tpr, color=mpl_color_list[color_idx], linestyle='solid', marker='', alpha=0.3, label=f'P{i+1} (AUC = {np.trapz(tpr, fpr):.2f})')
        else:
            plt.plot(fpr, tpr, color=mpl_color_list[color_idx], linestyle='solid', marker='', alpha=0.3, label=f'P{i+1}')

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(np.trapz(tpr, fpr))

    if chance_level:
        if auc:
            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=0.8, label='Chance (AUC = 0.50)')
        else:
            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=0.8, label='Chance')

    if mean:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.trapz(mean_tpr, mean_fpr)
        if auc:
            plt.plot(mean_fpr, mean_tpr, color='blue', lw=2, alpha=1, label=f'Mean ROC (AUC = {mean_auc:.2f})')
        else:
            plt.plot(mean_fpr, mean_tpr, color='blue', lw=2, alpha=1, label=f'Mean ROC')

    if std:
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label=r'$\pm$ 1 std dev')
    plt.title(title, fontsize=fontsize+2)
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=fontsize)
    plt.tight_layout()
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.savefig(save_path)
    plt.clf()


### モデルサイズ取得 ###
def get_model_memory_size(model):
    type_sizes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.float64: 8
    }
    total_memory_size = 0
    for p in model.parameters():
        total_memory_size += p.numel() * type_sizes[p.dtype]
    return total_memory_size

####################################################################################################









