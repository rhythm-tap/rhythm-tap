####################################################################################################
# 2入力の学習
####################################################################################################


from common_import import *

#---------------------------------------------------------------------------------------------------
# ハイパーパラメータ
data_dir = "../data_pro"
batch_size = 128
num_epochs = 10
validation_rate = 0.1
test_rate = 0.1
data_shuffle = True
learning_rate = 0.01
max_seq_len = 1500
# use_data=['acc','gyro']
use_data=['acc']
# use_data=['gyro']
model_structure = "lstm"
# model_structure = "transformer"


#---------------------------------------------------------------------------------------------------
# データ準備
batch_size = batch_size * GPU_COUNT
train_data, validation_data, test_data, pos_weight_dict = \
    makeCsvPathList(dir_path=data_dir, validation_rate=validation_rate, test_rate=test_rate)
train_dataset = WaveDataset(train_data, num_classes=2, max_seq_len=max_seq_len, use_data=use_data)
validation_dataset = WaveDataset(validation_data, num_classes=2, max_seq_len=max_seq_len, use_data=use_data)
test_dataset = WaveDataset(test_data, num_classes=2, max_seq_len=max_seq_len, use_data=use_data)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=data_shuffle)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=data_shuffle)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=data_shuffle)
print("Train data: "+str(len(train_dataset)))
print("Validation data: "+str(len(validation_dataset)))
print("Test data: "+str(len(test_dataset)))
print(pos_weight_dict)


#---------------------------------------------------------------------------------------------------
# モデル構築
if "lstm" == model_structure:
    input_dim = train_dataset.__getitem__(0)[0].shape[2] # 入力次元数
    hidden_dim = 256  # 隠れ層の次元数
    num_layers = 8  # LSTMの層数
    output_dim = 1  # 出力次元数
    model = LstmClassifier(input_dim, hidden_dim, num_layers, output_dim, max_seq_len)
elif "transformer" == model_structure:
    input_dim = train_dataset.__getitem__(0)[0].shape[2] # 入力次元数
    hidden_dim = 512  # モデルの埋め込み次元数
    num_layers = 8  # エンコーダ/デコーダの層数
    output_dim = 1  # 出力次元数
    nhead = 8  # マルチヘッド・アテンションのヘッド数
    model = TransformerClassifier(input_dim, hidden_dim, num_layers, output_dim, nhead)
else:
    print("Set 'model_structure' appropriately.")
    exit()
# torchsummary.summary(model.to('cpu'), tuple(train_dataset.__getitem__(0)[0].shape), device='cpu')
model = model.to(device)
print("MODEL SIZE: "+str(get_model_memory_size(model))+" (B)")


#---------------------------------------------------------------------------------------------------
# Multi GPU使用宣言
if str(device) == 'cuda':
    model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    print("Multi GPU OK.")
    print("\n")


#---------------------------------------------------------------------------------------------------
# 損失関数・最適化関数
pos_weight = torch.tensor(pos_weight_dict["train"], dtype=torch.float).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # pos_weight: 0ラベルのデータに対する1ラベルのデータの割合
optimizer = optim.Adam(model.module.parameters() if str(device) == 'cuda' else model.parameters(), lr = learning_rate)


#---------------------------------------------------------------------------------------------------
# メトリクス
metrics_dict = getMetrics(device, mode=['accuracy','auc','precision','recall','specificity','f1'], num_classes=2)


#---------------------------------------------------------------------------------------------------
# 学習ループ
history = {}
for epoch_idx in range(num_epochs):

    print_log(f"Epoch: {epoch_idx+1:>3}/{num_epochs}")

    train_history = train_one_epoch(model, device, train_dataloader, criterion, optimizer, metrics_dict=metrics_dict)
    validation_history = validation_one_epoch(model, device, validation_dataloader, criterion, metrics_dict=metrics_dict)
    
    for metrics, value in train_history.items():
        if metrics not in history:
            history[metrics] = []
        history[metrics].append(value)
    for metrics, value in validation_history.items():
        if metrics not in history:
            history[metrics] = []
        history[metrics].append(value)


#---------------------------------------------------------------------------------------------------
# モデル保存
save_path = "./model_checkpoint.pth"
torch.save({
    'epoch': num_epochs,
    'last_train_loss': train_history['loss'],
    'model_state_dict': model.module.state_dict() if str(device) == 'cuda' else model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)


#---------------------------------------------------------------------------------------------------
# グラフ描画
# loss描画
graph_data_loss = {'loss':history['loss']}
if 'val_loss' in history:
    graph_data_loss['val_loss'] = history['val_loss']
saveLossGraph(graph_data_loss, save_path='./loss.png', title='Model Loss (of this training)')

# accuracy描画
graph_data_acc = {}
if 'accuracy' in history:
    graph_data_acc['accuracy'] = history['accuracy']
if 'val_accuracy' in history:
    graph_data_acc['val_accuracy'] = history['val_accuracy']
if graph_data_acc != {}:
    saveLossGraph(graph_data_acc, save_path='./accuracy.png', title='Model Accuracy (of this training)')

# ROC曲線描画
# if 'roc' in test_history:
#     saveRocCurve(test_history['roc'], save_path=model_dir+'/roc.png', title='ROC Curve (of this training)')



print()
print()
print()