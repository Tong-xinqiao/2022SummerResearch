import numpy as np
import pandas as pd
import torch
from lassonet import LassoNetRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from DNN.data import portf_ret, sharpe
import time

"""SPECIFY TRAINING SET, VALIDATION SET AND TEST SET"""

time_start = time.time()
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
                 'OrgCap',
                 'RD_ME',
                 'Sale_ME', 'SUE_Chen', 'Turnover']

group_ind_dir = "../data/data_group_ind.csv"
r_eff_dir = "../data/data_R_eff.csv"
z_eff_dir = "../data/data_Z_eff.csv"
r_org_dir = "../data/data_R_org_eff.csv"

group_ind = pd.read_csv(group_ind_dir, header=None)
group_ind = group_ind.loc[:, 0]
r_eff = pd.read_csv(r_eff_dir, header=None)
z_eff = pd.read_csv(z_eff_dir, header=None)
r_org = pd.read_csv(r_org_dir, header=None)

t_train_start = list(range(1, 41 * 12 + 2, 12))
t_train_end = [x + 59 for x in t_train_start]
t_val_start = [x + 1 for x in t_train_end]
t_val_end = [x + 23 for x in t_val_start]
t_test_start = [x + 1 for x in t_val_end]
t_test_end = [x + 11 for x in t_test_start]
t_test_end[41] = 678

train_start = [int(group_ind[x - 1] + 1) for x in t_train_start]
train_end = [int(group_ind[x]) for x in t_train_end]
val_start = [int(group_ind[x - 1] + 1) for x in t_val_start]
val_end = [int(group_ind[x]) for x in t_val_end]
test_start = [int(group_ind[x - 1] + 1) for x in t_test_start]
test_end = [int(group_ind[x]) for x in t_test_end]

# portfolio_ret_test_all = []

retrain_idx = [39, 40]
for idx in retrain_idx:
    print('########' + str(idx) + '########')
    # training, validation and testing index
    train_idx = range(train_start[idx] - 1, train_end[idx])
    val_idx = range(val_start[idx] - 1, val_end[idx])
    test_idx = range(test_start[idx] - 1, test_end[idx])

    # data preparation
    X = z_eff.iloc[train_idx, :].to_numpy()
    Y = r_eff.iloc[train_idx].to_numpy()
    X_val = z_eff.iloc[val_idx, :].to_numpy()
    Y_val = r_eff.iloc[val_idx].to_numpy()
    X_test = z_eff.iloc[test_idx, :].to_numpy()
    Y_test = r_eff.iloc[test_idx].to_numpy()
    Y_org_test = r_org.iloc[test_idx].to_numpy()

    # check device for training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    # define the network
    """ 
        class lassonet.LassoNetRegressor(*, hidden_dims=(100), lambda_start='auto',
        lambda_seq=None, gamma=0.0, gamma_skip=0.0, path_multiplier=1.02, M=10,
        dropout=0, batch_size=None, optim=None, n_iters=(1000, 100), patience=(100, 10),
         tol=0.99, backtrack=False, val_size=None, device=None, verbose=1, random_state=None,
         torch_seed=None, class_weight=None, tie_approximation=None)
    """
    model = LassoNetRegressor(hidden_dims=(16, 8),
                              lambda_start=0.05, path_multiplier=1.02,
                              batch_size=64, verbose=True,
                              n_iters=(1000, 100), patience=(100, 10))
    path = model.path(X, Y, X_val=X_val, y_val=Y_val)

    # save feature selected under current idx
    fig = plt.figure(figsize=(12, 12))
    plt.grid(True)
    importances = model.feature_importances_.numpy()
    order = np.argsort(importances)[::-1]
    importances = importances[order]
    ordered_feature_names = [feature_names[i] for i in order]
    plt.bar(np.arange(63), importances)
    plt.xticks(np.arange(63), ordered_feature_names, rotation=90)
    plt.ylabel("Feature importance")
    plt.savefig("Selected_Feature" + str(idx) + ".png")
    plt.clf()

    n_selected = []
    mse = []
    # sharpe_n = []
    lambda_ = []
    selected_path = []
    for save in path:
        # save loss history under each lambda
        model.load(save.state_dict)
        y_pred = model.predict(X_test)
        n_selected.append(save.selected.sum().cpu().numpy())
        mse.append(mean_squared_error(Y_test, y_pred))
        # feature selected for each lambda
        lambda_.append(save.lambda_)
        selected_path.append(save.selected.cpu().numpy())
        # # monthly portfolio return under each lambda and sharpe ratio over the whole test time period
        # lambda_return_his = np.array([])
        # for t in range(t_test_start[idx], t_test_end[idx] + 1):
        #     start = group_ind[t - 1] - group_ind[t_test_start[idx] - 1]
        #     end = group_ind[t] - group_ind[t_test_start[idx] - 1]
        #     ret = portf_ret(start, end, y_pred, Y_org_test)
        #     lambda_return_his = np.append(lambda_return_his, ret)
        # if np.std(lambda_return_his) == 0:
        #     sharpe_n.append(0)
        # else:
        #     sharpe_n.append(np.mean(lambda_return_his) / np.std(lambda_return_his) * np.sqrt(12))

    # save selected features under each lambda
    df1 = pd.DataFrame()
    for i in range(len(lambda_)):
        df2 = pd.DataFrame()
        df2[lambda_[i]] = selected_path[i]
        df1 = pd.concat([df1, df2], axis=1)
    df1 = df1.set_index(feature_names)
    with pd.ExcelWriter("selected_path" + str(idx) + ".xlsx") as writer:
        df1.to_excel(writer, sheet_name="res")

    # save n_selected/mse for each idx
    plt.grid(True)
    plt.plot(n_selected, mse, ".-")
    plt.xlabel("number of selected features")
    plt.ylabel("Testing MSE")
    plt.savefig("LossHistory" + str(idx) + ".png")
    plt.clf()
    # # save n_selected/sharpe_n for each idx
    # plt.grid(True)
    # plt.plot(n_selected, sharpe_n, ".-")
    # plt.xlabel("number of selected features")
    # plt.ylabel("Sharpe ratio")
    # plt.savefig("SharpeHistory" + str(idx) + ".png")
    # plt.clf()

    # # choose the model whose lambda has least mse to save its return for this idx
    # chosen_model = path[np.argmin(mse)]
    # model.load(chosen_model.state_dict)
    # y_pred = model.predict(X_test)
    # for t in range(t_test_start[idx], t_test_end[idx] + 1):
    #     start = group_ind[t - 1] - group_ind[t_test_start[idx] - 1]
    #     end = group_ind[t] - group_ind[t_test_start[idx] - 1]
    #     ret = portf_ret(start, end, y_pred, Y_org_test)
    #     portfolio_ret_test_all.append(ret)

# # write output in csv file
# np.savetxt("portf_ret_nn_test_all.csv", portfolio_ret_test_all, delimiter=",", header="portf_ret_nn_test", fmt='%f')
time_end = time.time()
time_sum = time_end - time_start
print("Running time ", time_sum)
