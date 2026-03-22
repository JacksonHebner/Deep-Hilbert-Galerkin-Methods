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

#num_sample_points = 100
#adjuster = 1/(torch.arange(1,dim_in+1) ** 2)
#points = torch.randn(num_sample_points,dim_in) * adjuster
#torch.save(points, "points_500_dim.pt")


#points = torch.load("sample_points_stat_dist_500_dim.pt")
#xis = 2* torch.pi * torch.arange(0,251)/250
#fd_converter = torch.zeros(251, dim_in)
#for i in range(dim_in):
#    fd_converter[:,i] = torch.sin((i+1) / 2 * xis)
#fd_converter *= 1/math.sqrt(math.pi)

#points_fd = points @ fd_converter.T
#torch.save(points_fd, "sample_points_stat_dist_500_dim_fd.pt")

mc_evals = torch.load("mc_evals_deterministic_training_dist_500_dim.pt", map_location=torch.device('cpu'))
points = torch.load ("points_500_dim.pt")[:,range(100)]

model_evals = model(points).squeeze()

errs = model_evals - mc_evals
print("L^1/ME error:")
print(torch.abs(errs).mean().item())

print("L^2/RMSE error:")
print(torch.sqrt((errs**2).mean()).item())

print("Relative error 1:")
print(((torch.abs(errs) / torch.abs(mc_evals)).mean()).item())

print("Relative error 2:")
print(torch.sqrt((errs**2).sum() / (mc_evals**2).sum()).item())