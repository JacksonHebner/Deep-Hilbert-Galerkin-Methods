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
from neural_network_classes import SimpleNN

torch.set_default_device('cpu')

dim_in = 25
dim_cutoff_theoretical = 250

model = SimpleNN(dim_in=dim_in, num_neurons=600, num_layers=1, dim_out=1)
model.load_state_dict(torch.load("No control, no noise, 25 func, 250 hilbert, 600 neur one layer, WN stat.pt2", weights_only=True, map_location=torch.device('cpu')))
model.eval()

#for deterministic case
sigma = 0

#target dist is WN

eigenvals = (torch.arange(1,dim_cutoff_theoretical+1) ** 2)/4

def true_eval(sigma, point, model):
    return (sigma**2 / (1 + 2*eigenvals)).sum() + (point**2 / (1 + 2*eigenvals)).sum()
def model_eval(sigma, point, model):
    point_proj = point[0:dim_in]
    return model(point_proj.unsqueeze(0))
def stationary_dist_sample(dummy = 1):
    return torch.randn(dim_cutoff_theoretical) / torch.sqrt(2*eigenvals)
def target_dist_sample(dummy = 1):
    return torch.randn(dim_cutoff_theoretical) / torch.sqrt(2*eigenvals)

true_eval_vec = func.vmap(true_eval, in_dims=(None, 0, None))
model_eval_vec = func.vmap(model_eval, in_dims=(None, 0, None))
stationary_dist_sample_vec = func.vmap(stationary_dist_sample, in_dims=(0), randomness="different")
target_dist_sample_vec = func.vmap(target_dist_sample, in_dims=(0), randomness="different")

times = 1000000
points = target_dist_sample_vec(torch.zeros(times))
model_vals = model_eval_vec(sigma, points, model)
true_vals = true_eval_vec(sigma, points, model)
errors = model_vals - true_vals

print("L^1 error (ME):")
print(torch.abs(errors).mean().item())
print("L^2 error (RMSE):")
print((torch.sqrt((errors**2).mean())).item())
print("Relative error 1:")
print(((torch.abs(errors) / torch.abs(true_vals)).mean()).item())
print("Relative error 2:")
print(torch.sqrt((errors**2).sum() / (true_vals**2).sum()).item())