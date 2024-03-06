####################################################################################################
# 1入力の学習
####################################################################################################


from common_import import *

#---------------------------------------------------------------------------------------------------
# ハイパーパラメータ
data_dir = "../data"
batch_size = 64
num_epochs = 150
validation_rate = 0.1
test_rate = 0
data_shuffle = True
learning_rate = 0.001
max_seq_len = 1500
# use_data=['acc','gyro']
use_data=['acc']
# use_data=['gyro']
model_structure = "lstm"
# model_structure = "transformer"
personal_name = "user"

personal_name = sys.argv[1] if len(sys.argv)>1 else personal_name
data_dir = "../"+sys.argv[2] if len(sys.argv)>2 else data_dir

save_dir = f"./log_{personal_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"
os.makedirs(save_dir)
print(personal_name)

#---------------------------------------------------------------------------------------------------
# データ準備
batch_size = batch_size * GPU_COUNT
train_data, validation_data, test_data, pos_weight_dict = \
    makePersonalCsvPathList(personal_name, dir_path=data_dir, validation_rate=validation_rate, test_rate=test_rate, shuffle=True)
train_dataset = PersonalWaveDataset(train_data, num_classes=2, max_seq_len=max_seq_len, use_data=use_data)
validation_dataset = PersonalWaveDataset(validation_data, num_classes=2, max_seq_len=max_seq_len, use_data=use_data)
test_dataset = PersonalWaveDataset(test_data, num_classes=2, max_seq_len=max_seq_len, use_data=use_data)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=data_shuffle)
if len(validation_dataset)!=0:
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=data_shuffle)
else:
    validation_dataloader = None
if len(test_dataset)!=0:
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=data_shuffle)
else:
    test_dataloader = None
print("Train data: "+str(len(train_dataset)))
print("Validation data: "+str(len(validation_dataset)))
print("Test data: "+str(len(test_dataset)))
print(pos_weight_dict)


#---------------------------------------------------------------------------------------------------
# モデル構築
if "lstm" == model_structure:
    input_dim = train_dataset.__getitem__(0)[0].shape[1] # 入力次元数
    hidden_dim = 256  # 隠れ層の次元数
    num_layers = 10  # LSTMの層数
    output_dim = 1  # 出力次元数
    model = PersonalLstmClassifier(input_dim, hidden_dim, num_layers, output_dim, max_seq_len)
elif "transformer" == model_structure:
    input_dim = train_dataset.__getitem__(0)[0].shape[1] # 入力次元数
    hidden_dim = 512  # モデルの埋め込み次元数
    num_layers = 16  # エンコーダ/デコーダの層数
    output_dim = 1  # 出力次元数
    nhead = 8  # マルチヘッド・アテンションのヘッド数
    model = PersonalTransformerClassifier(input_dim, hidden_dim, num_layers, output_dim, nhead)
else:
    print("Set 'model_structure' appropriately.")
    exit()
torchinfo.summary(model)
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
momentum = (0.9, 0.999)
optimizer = optim.Adam(model.module.parameters() if str(device) == 'cuda' else model.parameters(), lr = learning_rate, betas=momentum)
step_size, gamma = 40, 0.1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


#---------------------------------------------------------------------------------------------------
# メトリクス
metrics_dict = getMetrics(device, mode=['accuracy','auc','precision','recall','specificity','f1'], num_classes=2)
metrics_list_test = ['accuracy','auc','precision','recall','specificity','f1','roc']
metrics_dict_test = getMetrics(device, mode=metrics_list_test, num_classes=2)


#---------------------------------------------------------------------------------------------------
# 条件出力
print(f"Epochs: {num_epochs}")
print(f"Batch Size: {batch_size}")
print(f"Learning Rate: {learning_rate}")
print(f"Validation Rate: {validation_rate}")
print(f"Test Rate: {test_rate}")
print(f"Data Shuffle: {data_shuffle}")
print(f"Max Sequence Length: {max_seq_len}")
print(f"Model Structure: {model_structure}")
print(f"Input Dim: {input_dim}")
print(f"Hidden Dim: {hidden_dim}")
print(f"Num Layers: {num_layers}")
print(f"Output Dim: {output_dim}")
print(f"Use Data: {use_data}")
print(f"Optimizer Momentum: {str(momentum)}")
print(f"Scheduler Step Size: {step_size}")
print(f"Scheduler Gamma: {gamma}")
print()


#---------------------------------------------------------------------------------------------------
# 学習ループ
history = {}
for epoch_idx in range(num_epochs):

    print_log(f"Epoch: {epoch_idx+1:>3}/{num_epochs}")

    train_history = train_one_epoch_personal(model, device, train_dataloader, criterion, optimizer, metrics_dict=metrics_dict)
    validation_history = validation_one_epoch_personal(model, device, validation_dataloader, criterion, metrics_dict=metrics_dict)
    
    for metrics, value in train_history.items():
        if metrics not in history:
            history[metrics] = []
        history[metrics].append(value)
    for metrics, value in validation_history.items():
        if metrics not in history:
            history[metrics] = []
        history[metrics].append(value)

    scheduler.step()

    if os.path.isfile(save_dir+'early_stop'):
        with open(save_dir+'early_stop', 'w') as f:
            f.write("Epoch Index = "+str(epoch_idx))
        print("Early Stopping: Epoch Index = "+str(epoch_idx))
        break


#---------------------------------------------------------------------------------------------------
# テスト表示
# test_dataloader = validation_dataloader
test_history = test(model, device, test_dataloader, criterion, metrics_dict=metrics_dict_test)
for m in metrics_list_test:
    if m != 'roc':
        print(f"Test {m.capitalize()}: {test_history[m]}")
print(test_history)
if 'roc' in test_history:
    #EER計算
    fpr, tpr, thresholds = test_history['roc']
    fpr, tpr, thresholds = fpr.cpu().numpy(), tpr.cpu().numpy(), thresholds.cpu().numpy()
    frr = 1 - tpr
    eer_index = np.argmin(np.abs(fpr - frr))
    eer = (fpr[eer_index] + frr[eer_index]) / 2
    test_history['eer'] = eer
    print(f"Test EER: {eer}")


#---------------------------------------------------------------------------------------------------
# モデル保存
save_path = save_dir+"model_checkpoint.pth"
torch.save({
    'epoch': num_epochs,
    'last_train_loss': train_history['loss'],
    'model_state_dict': model.module.state_dict() if str(device) == 'cuda' else model.state_dict(),
    'model_structure': model_structure,
    'input_dim': input_dim,
    'hidden_dim': hidden_dim,
    'num_layers': num_layers,
    'output_dim': output_dim,
    'nhead': nhead if 'nhead' in globals() else None,
    'max_seq_len': max_seq_len,
    'use_data': use_data,
    'optimizer_state_dict': optimizer.state_dict(),
}, save_path)


#---------------------------------------------------------------------------------------------------
# グラフ描画
# loss描画
graph_data_loss = {'loss':history['loss']}
if 'val_loss' in history:
    graph_data_loss['val_loss'] = history['val_loss']
saveLossGraph(graph_data_loss, save_path=save_dir+'loss.png', title='Model Loss (of this training)')

# accuracy描画
graph_data_acc = {}
if 'accuracy' in history:
    graph_data_acc['accuracy'] = history['accuracy']
if 'val_accuracy' in history:
    graph_data_acc['val_accuracy'] = history['val_accuracy']
if graph_data_acc != {}:
    saveLossGraph(graph_data_acc, save_path=save_dir+'accuracy.png', title='Model Accuracy (of this training)')

# ROC曲線描画
# if 'roc' in test_history:
#     saveRocCurve(test_history['roc'], save_path=model_dir+'/roc.png', title='ROC Curve (of this training)')



print("\n\n\n\n\n")
