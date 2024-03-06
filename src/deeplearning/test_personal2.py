####################################################################################################
# 2つのディレクトリからテストするコード
####################################################################################################

from common_import import *

#---------------------------------------------------------------------------------------------------
# ハイパーパラメータ
data_dir = "../data"  # テストデータのディレクトリを指定
batch_size = 64
base_dir = "/Users/user"

# personal_name = "aaa"
personal_name = ["aaa","bbb","ccc"]
log_name = "log_auth"


#---------------------------------------------------------------------------------------------------
# コマンドライン引数ロード
personal_name = sys.argv[1] if len(sys.argv)>1 else personal_name
log_name = sys.argv[2] if len(sys.argv)>2 else log_name


def test_personal(_personal_name, _log_name):

    #---------------------------------------------------------------------------------------------------
    # ディレクトリ整理
    print(f"Test Person: {_personal_name}")
    model_dir = ""
    for dname in os.listdir(base_dir+_log_name):
        if re.match(rf'^log_{_personal_name}_\d{{8}}_\d{{6}}$', dname):  # 正規表現でマッチングを確認
            model_dir = os.path.abspath(os.path.join(base_dir+_log_name, dname))
    if model_dir == "":
        print(f"Does not exist 'personal_name {_personal_name}'")
        exit()
    model_path = model_dir+"/model_checkpoint.pth"


    #---------------------------------------------------------------------------------------------------
    # チェックポイントロード
    checkpoint = torch.load(model_path)


    #---------------------------------------------------------------------------------------------------
    # テストデータの準備
    name_list.append(_personal_name)
    _, test_data, _, _ = makeFilteredPersonalCsvPathList(_personal_name, name_list, dir_path_1=data_dir1, dir_path_2=data_dir2, validation_rate=0.1, test_rate=0, shuffle=True)
    # test_dataset = PersonalWaveDataset(test_data, num_classes=2, max_seq_len=checkpoint['max_seq_len'], use_data=checkpoint['use_data'])
    test_dataset = PersonalWaveDataset(test_data, num_classes=2, max_seq_len=1500, use_data=['acc'])
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    print("Test data: "+str(len(test_dataset)))


    #---------------------------------------------------------------------------------------------------
    # モデルのロード
    # model_structure = checkpoint['model_structure']
    model_structure = "lstm"
    if "lstm" == model_structure:
        # model = PersonalLstmClassifier(checkpoint['input_dim'], checkpoint['hidden_dim'], checkpoint['num_layers'], checkpoint['output_dim'], max_seq_len)
        model = PersonalLstmClassifier(test_dataset.__getitem__(0)[0].shape[1], 256, 10, 1, 1500)
    elif "transformer" == model_structure:
        # model = PersonalTransformerClassifier(checkpoint['input_dim'], checkpoint['hidden_dim'], checkpoint['num_layers'], checkpoint['output_dim'], checkpoint['nhead'])
        model = PersonalTransformerClassifier(test_dataset.__getitem__(0)[0].shape[1], 512, 16, 1, 8)
    else:
        print("Unknown model structure.")
        exit()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    if str(device) == 'cuda':
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
    # モデルを評価モードに設定
    model.eval()


    #---------------------------------------------------------------------------------------------------
    # 損失関数・メトリクス
    criterion = nn.BCEWithLogitsLoss()
    metrics_list = ['accuracy','auc','precision','recall','specificity','f1','roc']
    metrics_dict = getMetrics(device, mode=metrics_list, num_classes=2)


    #---------------------------------------------------------------------------------------------------
    # テストループ
    test_history = test(model, device, test_dataloader, criterion, metrics_dict=metrics_dict)
    for m in metrics_list:
        if m != 'roc':
            print(f"Test {m.capitalize()}: {test_history[m]}")
    if 'roc' in test_history:
        #EER計算
        fpr, tpr, thresholds = test_history['roc']
        fpr, tpr, thresholds = fpr.cpu().numpy(), tpr.cpu().numpy(), thresholds.cpu().numpy()
        frr = 1 - tpr
        eer_index = np.argmin(np.abs(fpr - frr))
        eer = (fpr[eer_index] + frr[eer_index]) / 2
        test_history['eer'] = eer
        print(f"Test EER: {eer}")
    # print(test_history)


    #---------------------------------------------------------------------------------------------------
    # ROC曲線描画
    if 'roc' in test_history:
        saveRocCurve(test_history['roc'], save_path=model_dir+'/roc.png', title='ROC Curve (of this training)')



    print("\n\n\n\n\n")

    return test_history['roc'] if 'roc' in test_history else None



#---------------------------------------------------------------------------------------------------
# テスト
if isinstance(personal_name,list):
    roc_data_dict = {}
    for pn in personal_name:
        res_roc = test_personal(pn, log_name)
        if res_roc is not None:
            roc_data_dict[pn] = res_roc
    if roc_data_dict != {}:
        saveRocCurves(roc_data_dict, save_path=base_dir+log_name+'/rocs.png', title='ROC Curves', chance_level=True, mean=True, std=True)
else:
    test_personal(personal_name, log_name)
