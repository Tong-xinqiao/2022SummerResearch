import numpy as np
import torch
import torch.nn as nn
import pandas as pd

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
X = z_eff.iloc[train_idx, :]
Y = r_eff.iloc[train_idx]
X_val = z_eff.iloc[val_idx, :]
Y_val = r_eff.iloc[val_idx]


class Net(nn.Module):
    def __init__(self, num_hidden, hidden_dim, input_dim, output_dim):  # hidden_dim为list
        super(Net, self).__init__()
        self.num_hidden = num_hidden

        self.fc = nn.Linear(input_dim, hidden_dim[0])
        self.fc_list = []  # 存除了第一层

        for i in range(num_hidden - 1):
            self.fc_list.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.add_module('fc' + str(i + 2), self.fc_list[-1])
        self.fc_list.append(nn.Linear(hidden_dim[-1], output_dim))
        self.add_module('fc' + str(num_hidden + 1), self.fc_list[-1])

        self.prune_flag = 0
        self.mask = None

    def forward(self, x):
        if self.prune_flag == 1:
            for name, para in self.named_parameters():
                para.data[self.mask[name]] = 0

        x = torch.tanh(self.fc(x))
        for i in range(self.num_hidden - 1):
            x = torch.tanh(self.fc_list[i](x))
        x = self.fc_list[-1](x)
        return x  # 过一遍网络

    def set_prune(self, user_mask):
        self.mask = user_mask
        self.prune_flag = 1

    def cancel_prune(self):
        self.prune_flag = 0
        self.mask = None


def gradient(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False):
    """
    Compute the gradient of `outputs` with respect to `inputs`
    gradient(x.sum(), x)
    gradient((x * y).sum(), [x, y])
    """
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(outputs, inputs, grad_outputs,
                                allow_unused=True,
                                retain_graph=retain_graph,
                                create_graph=create_graph)  # 计算梯度
    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
    return torch.cat([x.contiguous().view(-1) for x in grads])  # 在torch里面，view函数相当于numpy的reshape
    # cat默认dim=0


def main():
    subn = 32
    prior_sigma_0 = 0.0001
    prior_sigma_1 = 0.01
    lambda_n = 0.03
    num_hidden = 1  # 3
    hidden_dim = [3]  # [6,4,3]
    num_seed = 10  # number of independent trials
    num_epoch = 10
    x_train, y_train, x_val, y_val = X, Y, X_val, Y_val
    output_dim = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    size = x_train.shape[0]
    input_dim = x_train.shape[1]  # 63

    for my_seed in range(num_seed):
        np.random.seed(my_seed)
        torch.manual_seed(my_seed)
        net = Net(num_hidden, hidden_dim, input_dim, output_dim)
        net.to(device)
        loss_func = nn.MSELoss()
        step_lr = 0.005  # 0.005
        momentum = 0  # 0
        optimization = torch.optim.SGD(net.parameters(), lr=step_lr, momentum=momentum)

        sigma = torch.FloatTensor([1]).to(device)
        c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(prior_sigma_0) - 0.5 * np.log(prior_sigma_1)
        c2 = 0.5 / prior_sigma_0 - 0.5 / prior_sigma_1
        threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(prior_sigma_1 / prior_sigma_0)) / (
                0.5 / prior_sigma_0 - 0.5 / prior_sigma_1))

        for epoch in range(num_epoch):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            shuffle_index = np.random.permutation(size)
            x_train = x_train.iloc[shuffle_index]
            y_train = y_train.iloc[shuffle_index]
            # train
            for iteration in range(size // subn):  # // 整除
                net.zero_grad()
                x_temp = x_train.iloc[(iteration * subn):((iteration + 1) * subn), :].to_numpy()
                x_tensor = torch.FloatTensor(x_temp).to(device)
                output = net(x_tensor)
                y_temp = y_train.iloc[(iteration * subn):((iteration + 1) * subn), :].to_numpy()
                y_tensor = torch.FloatTensor(y_temp).to(device)
                loss = loss_func(output, y_tensor)
                loss = loss.div(2 * sigma).add(sigma.log().mul(0.5))
                loss.backward()
                with torch.no_grad():
                    for para in net.parameters():
                        temp = para.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                        temp = para.div(-prior_sigma_0).mul(temp) + para.div(-prior_sigma_1).mul(1 - temp)
                        prior_grad = temp.div(size)
                        para.grad.data -= prior_grad
                optimization.step()
                if iteration % 100 == 0:
                    loss, current = loss.item(), (iteration + 1) * subn
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            # validation
            with torch.no_grad():
                x_val_tensor = torch.FloatTensor(x_val.to_numpy()).to(device)
                output = net(x_val_tensor)
                loss = loss_func(output, torch.FloatTensor(y_val.to_numpy()).to(device))
                loss = loss.div(2 * sigma).add(sigma.log().mul(0.5))
                print(f"Validation Error: {loss.item():>7f}")
        # set those weights less than threshold zero
        # user_mask = {}
        # for name, para in net.named_parameters():
        #     user_mask[name] = para.abs() < threshold
        # net.set_prune(user_mask)

        path = "model_para" + str(my_seed + 1) + ".pt"
        torch.save(net.state_dict(), path)


if __name__ == '__main__':
    main()
