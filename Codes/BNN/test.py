import torch
from torch.distributions import Normal
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib
import os


def sample(shape1, shape2):
    num = shape2 * shape1
    weights = [0.06, 0.97]
    sigmas = [0.1, 0.01]
    num_for_classes = np.random.multinomial(num, weights, size=1)[0]
    points = np.array([])
    for a, b in zip(num_for_classes, sigmas):
        if a == 0:
            continue
        sub_sample = Normal(0, b).sample([a])
        points = np.append(points, sub_sample.detach().numpy())

    random.shuffle(points)
    return points


def draw_pic(data):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文无法显示的问题
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.hist(sorted(data), bins=50)  # bins表示直方柱子数
    plt.ylabel('频数')
    plt.show()

# mean = sample(3, 63)
# rho = sample(3, 63)
# w_epsilon = Normal(0, 1).sample([3 * 63]).detach().numpy()
# weight = mean + np.log(1 + np.exp(rho)) * w_epsilon
# draw_pic(weight)
# draw_pic(mean)

