import pandas as pd
import numpy as np
from itertools import chain


# output the times the factors appear as significant in multiple independent trials
def identify(k, num_neuron, file_name, threshold):  # k independent trials
    dic = []
    useful_list = []    # contains significant factors in the k's trials
    use_cols = [i + 1 for i in range(num_neuron)]
    for i in range(k):
        significant_factors = []
        means = pd.read_excel(file_name, sheet_name=str(i + 1), header=1, usecols=use_cols)
        for j in range(means.shape[0]):
            means_line = abs(np.array(means.loc[j]))
            any_list = means_line > threshold
            if any(any_list):
                significant_factors.append(j + 1)
        useful_list.append(significant_factors)
    used_factors = []
    for m in range(len(useful_list)):
        for factor in useful_list[m]:
            if factor in used_factors:
                continue
            else:
                used_factors.append(factor)
                aim_list = list(chain.from_iterable(useful_list[m:]))
                dic.append(["factor " + str(factor), count_appear_times(aim_list, factor)])

    def by_index(li):
        return li[1]

    return sorted(dic, key=by_index, reverse=True), useful_list


def count_appear_times(list, ele):
    num = 0
    for i in list:
        if i == ele:
            num += 1
    return num


prior_sigma_0 = 0.0001
prior_sigma_1 = 0.01
lambda_n = 0.03
threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(prior_sigma_1 / prior_sigma_0)) / (
        0.5 / prior_sigma_0 - 0.5 / prior_sigma_1))

stat, useful_factors = identify(10, 3, "res3.xlsx", 0.1)
print(stat)
print(len(stat))


