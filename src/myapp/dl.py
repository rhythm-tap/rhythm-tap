
from myapp import app
import numpy as np
import torch
import sys
sys.path.append("../")
from deeplearning.common_import import *

# リアルタイム認証用
# 本人である確率を出力
def test_deeplearning(model_path, csv_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #---------------------------------------------------------------------------------------------------
    # チェックポイントロード
    checkpoint = torch.load(model_path, map_location=device)


    #---------------------------------------------------------------------------------------------------
    # テストデータの準備
    test_data = [[csv_path,1]]
    test_dataset = PersonalWaveDataset(test_data, num_classes=2, max_seq_len=checkpoint['max_seq_len'], use_data=checkpoint['use_data'])
    test_dataloader = DataLoader(test_dataset, batch_size=1)


    #---------------------------------------------------------------------------------------------------
    # モデルのロード
    model_structure = checkpoint['model_structure']
    if "lstm" == model_structure:
        model = PersonalLstmClassifier(checkpoint['input_dim'], checkpoint['hidden_dim'], checkpoint['num_layers'], checkpoint['output_dim'], checkpoint['max_seq_len'])
    elif "transformer" == model_structure:
        model = PersonalTransformerClassifier(checkpoint['input_dim'], checkpoint['hidden_dim'], checkpoint['num_layers'], checkpoint['output_dim'], checkpoint['nhead'])
    else:
        return False, "存在しないモデル構造を必要としています"
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    if str(device) == 'cuda':
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
    # モデルを評価モードに設定
    model.eval()

    #---------------------------------------------------------------------------------------------------
    # テストループ
    with torch.no_grad():
        for (inputs, labels) in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs)

    return probabilities.item()

