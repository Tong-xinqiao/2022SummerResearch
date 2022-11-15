from data import data_preparation, portf_ret, sharpe
from network import NeuralNetwork, model_train, model_val, prediction
import pandas as pd
import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim

group_ind_dir = "../data/data_group_ind.csv"
r_eff_dir = "../data/data_R_eff.csv"
z_eff_dir = "../data/data_Z_eff.csv"
r_org_dir = "../data/data_R_org_eff.csv"

group_ind = pd.read_csv(group_ind_dir, header=None)
group_ind = group_ind.loc[:, 0]
r_eff = pd.read_csv(r_eff_dir, header=None)
z_eff = pd.read_csv(z_eff_dir, header=None)
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
    print(idx)
    # training, validation and testing index
    train_idx = range(train_start[idx] - 1, train_end[idx])
    val_idx = range(val_start[idx] - 1, val_end[idx])
    test_idx = range(test_start[idx] - 1, test_end[idx])

    # data preparation
    train_set, val_set, test_set, r_org_train, r_org_val, r_org_test, input_size = data_preparation(z_eff, r_eff, r_org, train_idx, val_idx, test_idx, 32)

    # check device for training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    # define the network
    model = NeuralNetwork(input_size)

    # train the model
    epochs = 1
    loss_fn = nn.MSELoss()
    lr = 3e-4
    optimizer = optim.Adam(model.parameters(), lr)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        model_train(train_set, model, loss_fn, optimizer)
        model_val(val_set, model, loss_fn)

    # save model parameters
    # path = os.path.join("model_para_" + str(idx) + ".pt")
    # torch.save(model.state_dict(), path)

    # predictions
    preds_train = prediction(train_set, model)
    preds_val = prediction(val_set, model)
    preds_test = prediction(test_set, model)

    # print("preds_train", preds_train[0:50])
    # print("preds_val", preds_val[0:50])
    # print("preds_test", preds_test[0:50])

    # portfolio return
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

#     print("portf_ret_nn_train", portf_ret_nn_train)
#     print("portf_ret_nn_val", portf_ret_nn_val)
#     print("portf_ret_nn_test", portf_ret_nn_test)
#
#     # load model parameters
#     model = NeuralNetwork(input_size)
#     model.load_state_dict(torch.load(path))
#     model.eval()
#
#     for param_tensor in model.state_dict():
#         print(param_tensor, "\t", model.state_dict()[param_tensor])
#
# # sharpe ratio
# sharpe_nn_train = sharpe(portf_ret_nn_train[0:t_train_end[41]])
# sharpe_nn_val = sharpe(portf_ret_nn_val[120:t_val_end[41]])
# sharpe_nn_test = sharpe(portf_ret_nn_test[180:t_test_end[41]])
#
# print("sharpe_nn_train", sharpe_nn_train)
# print("sharpe_nn_val", sharpe_nn_val)
# print("sharpe_nn_test", sharpe_nn_test)
#
# write output in csv file
np.savetxt("portf_ret_nn_train.csv", portf_ret_nn_train, delimiter=",", header="portf_ret_nn_train", fmt='%f')
np.savetxt("portf_ret_nn_val.csv", portf_ret_nn_val, delimiter=",", header="portf_ret_nn_val", fmt='%f')
np.savetxt("portf_ret_nn_test.csv", portf_ret_nn_test, delimiter=",", header="portf_ret_nn_test", fmt='%f')
#
# np.savetxt("sharpe_nn_train.csv", sharpe_nn_train, delimiter=",", header="sharpe_nn_train", fmt='%f')
# np.savetxt("sharpe_nn_val.csv", sharpe_nn_val, delimiter=",", header="sharpe_nn_val", fmt='%f')
# np.savetxt("sharpe_nn_test.csv", sharpe_nn_test, delimiter=",", header="sharpe_nn_test", fmt='%f')
