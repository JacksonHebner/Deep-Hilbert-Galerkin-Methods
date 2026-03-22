import numpy as np
import torch
from torch.func import vmap, grad, functional_call
import torch.func as func
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm, trange
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from neural_network_classes import SimpleNN, DGMNet

torch.set_default_device('cpu')

dim_in = 100

#model = SimpleNN(dim_in=dim_in, num_neurons=1000, num_layers=1, dim_out=1)
model = DGMNet(in_dim=dim_in, n_layers = 3)
model.load_state_dict(torch.load("sigma = 0, 100 func, 500 hilbert, DGMNet, TCC tail, QPDE fixed.pt2", weights_only=True, map_location=torch.device('cpu')))
model.eval()

print("Model at x = 0")
test_1 = torch.zeros(dim_in)
print(model(test_1.unsqueeze(0)).item())

print("Model at x = sin(xi)/sqrt(pi)")
test_2 = torch.zeros(dim_in)
test_2[1] = 1
print(model(test_2.unsqueeze(0)).item())

print("Model at x = sin(2xi)/sqrt(pi)")
test_3 = torch.zeros(dim_in)
test_3[3] = 1
print(model(test_3.unsqueeze(0)).item())

print("Model at x = sin(3xi)/sqrt(pi)")
test_4 = torch.zeros(dim_in)
test_4[5] = 1
print(model(test_4.unsqueeze(0)).item())

print("Model at x = xi/2pi * (2pi-xi)")
k_int = torch.arange(1, dim_in+1)
odd = (k_int % 2 == 1)
test_5 = 16.0 * odd / (k_int**3 * (math.pi ** 1.5))
print(model(test_5.unsqueeze(0)).item())

print("Model at x = 1 - cos(xi)")
n = torch.arange(1, dim_in + 1, dtype=torch.get_default_dtype())
odd = (n % 2 == 1)
test_6 = torch.zeros_like(n)
test_6[odd] = -16.0 / ((n[odd] - 2) * n[odd] * (n[odd] + 2) * math.sqrt(math.pi))
print(model(test_6.unsqueeze(0)).item())

print("Model at x = 1 - cos(2xi)")
n = torch.arange(1, dim_in + 1, dtype=torch.get_default_dtype())
odd = (n % 2 == 1)
test_7 = torch.zeros_like(n)
test_7[odd] = -64.0 / ((n[odd] - 4) * n[odd] * (n[odd] + 4) * math.sqrt(math.pi))
print(model(test_7.unsqueeze(0)).item())

print("Model at x = 1/sqrt(2pi)")
k_int = torch.arange(1, dim_in+1)
odd = (k_int % 2 == 1)
test_8 = (4.0/torch.pi) * odd / k_int / math.sqrt(2)
print(model(test_8.unsqueeze(0)).item())