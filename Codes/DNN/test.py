import pandas as pd

t_train_start = list(range(1, 41 * 12 + 2, 12))  # len = 42
t_train_end = [x + 119 for x in t_train_start]
t_val_start = [x + 1 for x in t_train_end]
t_val_end = [x + 59 for x in t_val_start]
t_test_start = [x + 1 for x in t_val_end]
t_test_end = [x + 11 for x in t_test_start]
t_test_end[41] = 678
group_ind_dir = "../data/data_group_ind.csv"
group_ind = pd.read_csv(group_ind_dir, header=None)
group_ind = group_ind.loc[:, 0]
train_start = [int(group_ind[x - 1] + 1) for x in t_train_start]
train_end = [int(group_ind[x]) for x in t_train_end]
val_start = [int(group_ind[x - 1] + 1) for x in t_val_start]
val_end = [int(group_ind[x]) for x in t_val_end]
test_start = [int(group_ind[x - 1] + 1) for x in t_test_start]
test_end = [int(group_ind[x]) for x in t_test_end]

idx = 2
for t in range(t_train_start[idx], t_train_end[idx] + 1):
    start = group_ind[t - 1] - group_ind[t_train_start[idx] - 1]
    end = group_ind[t] - group_ind[t_train_start[idx] - 1]
    print(start, end, t - 1)
