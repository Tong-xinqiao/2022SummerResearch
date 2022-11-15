from DNN.data import data_preparation
from MixPriorBNN import BNN, model_train, model_val
import pandas as pd
import torch
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
t_train_end = [x + 11 for x in t_train_start]  # 119
t_val_start = [x + 1 for x in t_train_end]
t_val_end = [x + 5 for x in t_val_start]  # 59
t_test_start = [x + 1 for x in t_val_end]
t_test_end = [x + 11 for x in t_test_start]
t_test_end[41] = 678  # 多少个月

train_start = [int(group_ind[x - 1] + 1) for x in t_train_start]
train_end = [int(group_ind[x]) for x in t_train_end]
val_start = [int(group_ind[x - 1] + 1) for x in t_val_start]
val_end = [int(group_ind[x]) for x in t_val_end]
test_start = [int(group_ind[x - 1] + 1) for x in t_test_start]
test_end = [int(group_ind[x]) for x in t_test_end]

idx = 41
train_idx = range(train_start[idx] - 1, train_end[idx])
val_idx = range(val_start[idx] - 1, val_end[idx])
test_idx = range(test_start[idx] - 1, test_end[idx])

# data preparation
train_set, val_set, test_set, r_org_train, \
r_org_val, r_org_test, input_size = data_preparation(z_eff, r_eff, r_org, train_idx, val_idx, test_idx, 32)


def multi_trials(k):
    # check device for training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    # define the network
    learning_rate = 0.03
    model = BNN(input_size, 3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # train the model
    epochs = 10
    fine_tune_epochs = 10
    threshold = 0.5

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        model_train(train_set, model, optimizer)
        model_val(val_set, model)

    # user_mask = {}
    # for name, para in model.named_parameters():
    #     user_mask[name] = para.abs() < threshold
    #
    # for name, para in model.named_parameters():
    #     para.data[user_mask[name]] = 0

    path1 = "model_para_prior" + str(k) + ".pt"
    torch.save(model.state_dict(), path1)

    # model.set_prune(user_mask)

    # for fine_tune_epoch in range(fine_tune_epochs):
    #     print(f"fine_tune_epoch {fine_tune_epoch + 1}\n-------------------------------")
    #     model_train(train_set, model, optimizer)
    #     model_val(val_set, model)

    # user_mask = {}
    # for name, para in model.named_parameters():
    #     user_mask[name] = para.abs() < threshold
    #
    # for name, para in model.named_parameters():
    #     para.data[user_mask[name]] = 0

    # path2 = "model_para_post5.pt"
    # torch.save(model.state_dict(), path2)


for i in range(1):
    multi_trials(i + 1)
