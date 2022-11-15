import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


def dataset_partition(x, y, y_org, train_idx, val_idx, test_idx):
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    y_org = pd.DataFrame(y_org)
    x_train = x.loc[train_idx, :]

    # remove constant variables in train set
    x_train = x_train.loc[:, x_train.var() > 0]  # 方差不为0的所有列
    x_idx = list(x_train.columns)
    x_num_col = len(x_train.columns)

    # dataset partition
    y_train = y.loc[train_idx, :]
    y_val = y.loc[val_idx, :]
    y_test = y.loc[test_idx, :]
    y_org_train = y_org.loc[train_idx, :]
    y_org_val = y_org.loc[val_idx, :]
    y_org_test = y_org.loc[test_idx, :]
    x_val = x.loc[val_idx, x_idx]
    x_test = x.loc[test_idx, x_idx]

    # normalize x
    x_train = (x_train - x_train.mean())/x_train.std()
    x_val = (x_val - x_val.mean())/x_val.std()
    x_test = (x_test - x_test.mean())/x_test.std()

    return x_train, y_train, y_org_train, x_val, y_val, y_org_val, x_test, y_test, y_org_test, x_num_col


# create a map-style dataset
class Data(Dataset):
    def __init__(self, y, x):
        self.y = torch.from_numpy(np.array(y.astype('float32')))
        self.x = torch.from_numpy(np.array(x.astype('float32')))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        y = self.y
        x = self.x
        return y[idx], x[idx]


def data_preparation(x, y, y_org, train_idx, val_idx, test_idx, batch_size):
    x_train, y_train, y_org_train, x_val, y_val, y_org_val, \
    x_test, y_test, y_org_test, input_size = dataset_partition(x, y, y_org, train_idx, val_idx, test_idx)
    train = Data(y_train, x_train)
    val = Data(y_val, x_val)
    test = Data(y_test, x_test)

    # prepare the data loaders
    train_set = DataLoader(train, batch_size)
    val_set = DataLoader(val, batch_size)
    test_set = DataLoader(test, batch_size)
    return train_set, val_set, test_set, y_org_train, y_org_val, y_org_test, input_size


def portf_ret(start, end, preds, y_org):
    y_org = np.array(y_org).flatten()
    y_hat_t = preds[start:end]
    y_true_t = y_org[start:end]
    list_buy = np.argwhere(y_hat_t >= np.quantile(y_hat_t, 0.9))
    list_sell = np.argwhere(y_hat_t <= np.quantile(y_hat_t, 0.1))
    portf_ret_t = np.mean(y_true_t[list_buy]) - np.mean(y_true_t[list_sell])
    return portf_ret_t


def sharpe(returns, riskless=0):
    returns = np.array(returns)
    riskless = np.array(riskless)
    std = np.std(returns)
    sharpe = (returns - riskless)/std
    return sharpe

