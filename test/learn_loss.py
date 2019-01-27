import torch
import torch.nn as nn

torch.manual_seed(123)

# learn BCEWithLogitsLoss to see should I reshape input and label
# Result: I don't need.
N, T = 2, 3
stop_pred = torch.rand(N, T)
stop_target = torch.randint(2, (N, T))

stop_loss = nn.BCEWithLogitsLoss()(stop_pred, stop_target)
print(stop_pred)
print(stop_target)
print(stop_loss)

stop_pred = stop_pred.view(-1, 1)
stop_target = stop_target.view(-1, 1)
stop_loss = nn.BCEWithLogitsLoss()(stop_pred, stop_target)
print(stop_pred)
print(stop_target)
print(stop_loss)


def sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))


# learn MSELoss to see should I reshape input and label
# Result: I don't need.
N, T, D = 2, 3, 2
pred = torch.ones(N, T, D)
label = torch.zeros(N, T, D)
loss = nn.MSELoss()(pred, label)
print(pred)
print(label)
print(loss)
