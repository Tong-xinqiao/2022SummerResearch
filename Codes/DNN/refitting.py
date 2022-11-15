from data import data_preparation, portf_ret, sharpe
from network import NeuralNetwork, model_train, prediction
import pandas as pd
import torch.nn as nn
import torch
import numpy as np
from early_stopping import EarlyStopping
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import random

group_ind_dir = "../data/data_group_ind.csv"
r_eff_dir = "../data/data_R_eff.csv"
z_eff_dir = "../data/data_Z_eff.csv"
r_org_dir = "../data/data_R_org_eff.csv"

feature_names = ['Accrual', 'BtM_GP', 'F_Score', 'GPA', 'NI_ReA', 'BM_GP_Mom', 'BM_Mom', 'FP', 'Ivol',
                 'Mom', 'NI_ReM', 'ROE', 'SUE', 'IRR', 'Season', "???", 'Beta', 'BidAsk_Spread',
                 'Days_ZeroTrade', 'Div',
                 'Dol_Volume', 'DolVolume_ME', 'IndMom_Size3', 'Max_Ret', 'Mom_0612', 'Mom_1318',
                 'Mom0206_volume5',
                 'Price', 'Size_Chen', 'Turnover_Vol', 'Volume_Trend', 'Volume_Vol', 'WeekHigh52', 'Abr_1m',
                 'AssetGrowth_Chen',
                 'AssetTurnover', 'BtM_Chen', 'CashProd', 'ConvDebt', 'DelCAPEX', 'DelEmp', 'DelEq',
                 'DelGM_DelSale',
                 'DelInvt', 'DelLTNOA', 'DelMom', 'DelPPE_Invt', 'DelSale', 'DelSale_DelAR', 'DelSale_DelInvt',
                 'DelSale_DelXSGA',
                 'DelSO', 'DelTax', 'Illiquidity', 'Ind_Mom_Chen', 'Ind_Mom0206', 'Ind_Mom0212', 'NumCEI',
                 'OrgCap', 'RD_ME', 'Sale_ME', 'SUE_Chen', 'Turnover']

# used_features = ['DelCAPEX', 'Div', 'Dol_Volume', 'IRR', 'Days_ZeroTrade', 'Ind_Mom0212', 'AssetGrowth_Chen',
#                  'Price', 'SUE_Chen', 'Ind_Mom_Chen', 'F_Score', 'Ivol', 'Mom', 'NumCEI',
#                  'Sale_ME']  # [2,8,9,13,18,19,20,27,34,39,54,56,57,60,61]
#
# used_cols = sorted([feature_names.index(i) for i in used_features])
used_cols = sorted(random.sample(range(63), 15))
print(used_cols)
group_ind = pd.read_csv(group_ind_dir, header=None)
group_ind = group_ind.loc[:, 0]
r_eff = pd.read_csv(r_eff_dir, header=None)
z_eff = pd.read_csv(z_eff_dir, header=None, usecols=used_cols)
r_org = pd.read_csv(r_org_dir, header=None)

t_train_start = list(range(1, 41 * 12 + 2, 12))  # len = 42
t_train_end = [x + 119 for x in t_train_start]
t_val_start = [x + 1 for x in t_train_end]
t_val_end = [x + 59 for x in t_val_start]
t_test_start = [x + 1 for x in t_val_end]
t_test_end = [x + 11 for x in t_test_start]
t_test_end[41] = 678

train_start = [int(group_ind[x - 1] + 1) for x in t_train_start]
train_end = [int(group_ind[x]) for x in t_train_end]
val_start = [int(group_ind[x - 1] + 1) for x in t_val_start]
val_end = [int(group_ind[x]) for x in t_val_end]
test_start = [int(group_ind[x - 1] + 1) for x in t_test_start]
test_end = [int(group_ind[x]) for x in t_test_end]

portf_ret_nn_train = np.zeros(t_train_end[41])
portf_ret_nn_val = np.zeros(t_val_end[41])
portf_ret_nn_test = np.zeros(t_test_end[41])

retrain_idx = list(range(0, 42))
for idx in retrain_idx:
    print("##########" + str(idx) + "##########")
    train_idx = range(train_start[idx] - 1, train_end[idx])
    val_idx = range(val_start[idx] - 1, val_end[idx])
    test_idx = range(test_start[idx] - 1, test_end[idx])
    # data preparation
    train_set, val_set, test_set, r_org_train, \
    r_org_val, r_org_test, input_size = data_preparation(z_eff, r_eff, r_org,
                                                         train_idx, val_idx,
                                                         test_idx, 32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    model = NeuralNetwork(input_size)
    epochs = 100
    loss_fn = nn.MSELoss()
    lr = 3e-4
    optimizer = optim.Adam(model.parameters(), lr)
    save_path = ".\\"
    early_stopping = EarlyStopping(save_path, patience=10)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        model_train(train_set, model, loss_fn, optimizer)
        # validation
        num_batches = len(val_set)
        val_loss = 0
        with torch.no_grad():
            for batch, (y, x) in enumerate(val_set):
                pred = model(x)
                val_loss += loss_fn(pred, y).item()
        val_loss /= num_batches
        print(f"Validation Error: \n Avg loss: {val_loss:>8f} \n")
        # early stop
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    preds_train = prediction(train_set, model)
    preds_val = prediction(val_set, model)
    preds_test = prediction(test_set, model)
    # Y_test = r_eff.iloc[test_idx].to_numpy()
    # loss = mean_squared_error(preds_test, Y_test)

    for t in range(t_train_start[idx], t_train_end[idx] + 1):
        start = group_ind[t - 1] - group_ind[t_train_start[idx] - 1]
        end = group_ind[t] - group_ind[t_train_start[idx] - 1]
        portf_ret_nn_train[t - 1] = portf_ret(start, end, preds_train, r_org_train)

    for t in range(t_val_start[idx], t_val_end[idx] + 1):
        start = group_ind[t - 1] - group_ind[t_val_start[idx] - 1]
        end = group_ind[t] - group_ind[t_val_start[idx] - 1]
        portf_ret_nn_val[t - 1] = portf_ret(start, end, preds_val, r_org_val)
    for t in range(t_test_start[idx], t_test_end[idx] + 1):
        start = group_ind[t - 1] - group_ind[t_test_start[idx] - 1]
        end = group_ind[t] - group_ind[t_test_start[idx] - 1]
        portf_ret_nn_test[t - 1] = portf_ret(start, end, preds_test, r_org_test)

np.savetxt("portf_ret_nn_train.csv", portf_ret_nn_train, delimiter=",", header="portf_ret_nn_train", fmt='%f')
np.savetxt("portf_ret_nn_val.csv", portf_ret_nn_val, delimiter=",", header="portf_ret_nn_val", fmt='%f')
np.savetxt("portf_ret_nn_test.csv", portf_ret_nn_test, delimiter=",", header="portf_ret_nn_test", fmt='%f')

