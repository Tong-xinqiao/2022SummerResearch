import torch
import torch.nn as nn
import numpy as np


# define the model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.linear_relu = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        y_hat = self.linear_relu(x)
        return y_hat


# model training
def model_train(train_set, model, loss_fn, optimizer):
    size = len(train_set.dataset)
    for batch, (y, x) in enumerate(train_set):
        # Compute prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # if batch == 100:
            #     return


# model validation
def model_val(val_set, model, loss_fn):
    num_batches = len(val_set)
    val_loss = 0

    with torch.no_grad():
        for batch, (y, x) in enumerate(val_set):
            pred = model(x)
            val_loss += loss_fn(pred, y).item()
            # if batch == 100:
            #     break

    val_loss /= num_batches
    print(f"Validation Error: \n Avg loss: {val_loss:>8f} \n")


def prediction(dataset, model):
    pred = np.array([])
    for batch, (y, x) in enumerate(dataset):
        y_hat = model(x)
        y_hat = y_hat.detach().numpy()
        y_hat = y_hat.reshape((len(y_hat), 1))
        pred = np.append(pred, y_hat)
        # if batch % 100 == 0:
        #     print("pred", pred[max(0, batch-100):50])
        #     if batch == 100:
        #         break
    return pred
