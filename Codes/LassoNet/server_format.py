import numpy as np
import pandas as pd
import torch
from lassonet import LassoNetRegressor
import matplotlib.pyplot as plt

feature_names = np.array(['Accrual', 'BtM_GP', 'F_Score', 'GPA', 'NI_ReA', 'BM_GP_Mom', 'BM_Mom', 'FP', 'Ivol',
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
                          'Sale_ME', 'SUE_Chen', 'Turnover'])

group_ind_dir = "../data/data_group_ind.csv"
r_eff_dir = "../data/data_R_eff.csv"
z_eff_dir = "../data/data_Z_eff.csv"

group_ind = pd.read_csv(group_ind_dir, header=None)
group_ind = group_ind.loc[:, 0]
r_eff = pd.read_csv(r_eff_dir, header=None)
z_eff = pd.read_csv(z_eff_dir, header=None)

t_train_start = list(range(1, 56 * 12, 12))  # len = 56
t_train_end = [x + 11 for x in t_train_start]  # one year training and validation
t_train_end[-1] = 678

train_start = [int(group_ind[x - 1] + 1) for x in t_train_start]
train_end = [int(group_ind[x]) for x in t_train_end]

idx = 40  # [0,1,2,...,55]
print('########' + str(idx) + '########')
train_idx = range(train_start[idx] - 1, train_end[idx])

# data preparation
X = z_eff.iloc[train_idx, :].to_numpy()
Y = r_eff.iloc[train_idx].to_numpy()

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
                          batch_size=32, verbose=True,
                          n_iters=(1000, 100), patience=(100, 10))
path = model.path(X, Y)

# save feature selected under current idx
fig = plt.figure(figsize=(12, 12))
plt.grid(True)
importances = model.feature_importances_.numpy()
order = np.argsort(importances)[::-1]
importances = importances[order]
ordered_feature_names = [feature_names[i] for i in order]
plt.bar(np.arange(63), importances)
plt.xticks(np.arange(63), ordered_feature_names, rotation=90)
plt.ylabel("Feature Importance")
plt.savefig("Selected_Feature" + str(idx) + ".png")
plt.clf()
df = pd.DataFrame({"Feature names": ordered_feature_names, "Importance": importances})
with pd.ExcelWriter("Feature_importance" + str(idx) + ".xlsx") as writer:
    df.to_excel(writer, sheet_name="res")

n_selected = []
lambda_ = []
selected_path = []
for save in path:
    n_selected.append(save.selected.sum().cpu().numpy())
    # feature selected for each lambda
    lambda_.append(save.lambda_)
    selected_path.append(save.selected.cpu().numpy())

# save selected features under each lambda
df1 = pd.DataFrame()
for i in range(len(lambda_)):
    df2 = pd.DataFrame()
    df2[lambda_[i]] = selected_path[i]
    df1 = pd.concat([df1, df2], axis=1)
df1 = df1.set_index(feature_names)
with pd.ExcelWriter("selected_path" + str(idx) + ".xlsx") as writer:
    df1.to_excel(writer, sheet_name="res")
