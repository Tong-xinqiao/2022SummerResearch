import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random


class ScaleMixtureGaussian(object):
    def __init__(self, lambda_, sigma1, sigma2):
        super().__init__()
        self.lambda_ = lambda_
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def sample(self, shape1, shape2):
        num = shape1 * shape2
        weights = [self.lambda_, 1 - self.lambda_]
        sigmas = [self.sigma1, self.sigma2]
        num_for_classes = np.random.multinomial(num, weights, size=1)[0]
        points = np.array([])
        for a, b in zip(num_for_classes, sigmas):
            if a == 0:
                continue
            sub_sample = Normal(loc=0, scale=b).sample([a])
            points = np.append(points, sub_sample.detach().numpy())
        random.shuffle(points)
        return torch.from_numpy(points.reshape(shape1, shape2))


lamda = 0.03
SIGMA_1 = 0.1
SIGMA_2 = 0.003


class Layer(nn.Module):
    def __init__(self, input_features, output_features):
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(ScaleMixtureGaussian(lamda, SIGMA_1, SIGMA_2).sample(output_features, input_features))
        self.w_rho = nn.Parameter(ScaleMixtureGaussian(lamda, SIGMA_1, SIGMA_2).sample(output_features, input_features))
        self.weight = None

        # initialize mu and rho parameters for the layer's bias
        self.b_mu = nn.Parameter(ScaleMixtureGaussian(lamda, SIGMA_1, SIGMA_2).sample(1, output_features)[0])
        self.b_rho = nn.Parameter(ScaleMixtureGaussian(lamda, SIGMA_1, SIGMA_2).sample(1, output_features)[0])
        self.bias = None

    def forward(self, input):
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape)
        self.weight = (self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon).float()
        b_epsilon = Normal(0, 1).sample(self.b_mu.shape)
        self.bias = (self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * b_epsilon).float()
        return F.linear(input, self.weight, self.bias)


class BNN(nn.Module):
    def __init__(self, input_size, hidden_units):
        super().__init__()
        self.hidden = Layer(input_size, hidden_units)
        self.out = Layer(hidden_units, 1)

        self.prune_flag = 0
        self.mask = None

    def forward(self, x):
        if self.prune_flag == 1:
            for name, para in self.named_parameters():
                para.data[self.mask[name]] = 0

        x = torch.relu(self.hidden(x))
        x = self.out(x)
        return x

    def set_prune(self, user_mask):
        self.mask = user_mask
        self.prune_flag = 1

    def loss_fun(self, x, target):
        sigma = torch.FloatTensor([1])
        lf = nn.MSELoss()
        output = self(x)
        loss = lf(output, target)
        loss = loss.div(2 * sigma).add(sigma.log().mul(0.5))
        return loss


def model_train(train_set, model, optimizer):
    c1 = np.log(lamda) - np.log(1 - lamda) + 0.5 * np.log(SIGMA_1) - 0.5 * np.log(SIGMA_2)
    c2 = 0.5 / SIGMA_1 - 0.5 / SIGMA_2
    size = len(train_set.dataset)
    for batch, (y, x) in enumerate(train_set):
        loss = model.loss_fun(x, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            for para in model.parameters():
                temp = para.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                temp = para.div(-SIGMA_1).mul(temp) + para.div(-SIGMA_2).mul(1 - temp)
                prior_grad = temp.div(size)
                para.grad.data -= prior_grad
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def model_val(val_set, model):
    num_batches = len(val_set)
    val_loss = 0
    with torch.no_grad():
        for batch, (y, x) in enumerate(val_set):
            val_loss += model.loss_fun(x, y).item()
    val_loss /= num_batches
    print(f"Validation Error: \n Avg loss: {val_loss:>8f} \n")
