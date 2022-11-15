import numpy as np
import torch
import argparse

# # (1) 声明一个parser
# parser = argparse.ArgumentParser()
# # (2) 添加参数
# parser.add_argument("parg")  # 位置参数，这里表示第一个出现的参数赋值给parg
# parser.add_argument("--digit", type=int, help="输入数字")  # 通过 --echo xxx声明的参数，为int类型
# parser.add_argument("--name", help="名字", default="cjf")  # 同上，default 表示默认值
# # (3) 读取命令行参数
# args = parser.parse_args()
#
# # (4) 调用这些参数
# print(args.parg)
# print("echo ={0}".format(args.digit))
# print("name = {}".format(args.name))


# backward, autograd.grad
# https://zhuanlan.zhihu.com/p/279758736

# x = torch.tensor(2., requires_grad=True)
# a = torch.add(x, 1)
# b = torch.add(x, 2)
# y = torch.mul(a, b)
# grad = torch.autograd.grad(outputs=y, inputs=x)
# print(grad[0])
#
# x = torch.tensor(2., requires_grad=True)
# y = torch.tensor(3., requires_grad=True)
# z = x * x * y
# z.backward()
# print(x.grad, y.grad)

# zip
# x = [1, 2, 3, 4, 5]
# y = ['a', 'b', 'c', 'd']
# for a, b in zip(x, y):
#     print(a, b)

# x = [[1, 2, 3, 4], ['a', 'b', 'c'], [7, 8, 9]]
# y = zip(*x)
# for a in y:
#     print(a)
# for a, b, c in zip(*x):
#     print(a, b, c)

# x = torch.tensor([1., 2., 3, 4, 5, 6])
# print(x.contiguous().view(-1, 2))


# x1 = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.int)
# x2 = torch.tensor([[11, 22, 33], [22, 33, 44]], dtype=torch.int)
# inputs = [x1, x2]
# print(torch.cat(inputs, dim=0))
# print(torch.cat(inputs, dim=1))

