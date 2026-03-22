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

torch.set_default_device('cuda')

alpha = 1 #this simplified problem is the 1D stochastic Burgers equation, so Af = \alpha f'' + ff'
gamma  = 1
sigma = 1
lam = 1

#time_limit = 1 #i.e. T, the upper bound of time interval [0,T]
lower_bound = 0 #i.e. the L in [L,U]
upper_bound = 2 * torch.pi

#number of points
basis_size = 250
basis_increment = 2*torch.pi / basis_size

time_steps = 100000
time_increment = 0.0001

#control for now is uniform
control = 0 #torch.sin(torch.arange(0, basis_size+1) * 2 * torch.pi / basis_size)

#target for now is uniform
target = 0 #torch.sin(torch.arange(0, basis_size+1) * 2 * torch.pi / basis_size)

#noise vector
noise = torch.ones(basis_size+1) / math.sqrt(2*math.pi)

def finite_differences(mc): #takes in (basis_size+1,) tensor of initial values obeying the zero endpoint boundary condition
    cost = 0
    rectangle_size = time_increment * basis_increment
    for i in range(time_steps - 1):
        time_discount = math.exp(-gamma * i * time_increment)

        first_deriv = (mc.roll(-1) - mc)/basis_increment
        first_deriv[basis_size] = first_deriv[basis_size-1]
        second_deriv = (mc.roll(-1) - 2 * mc + mc.roll(1))/basis_increment**2

        diff = alpha*second_deriv + mc*first_deriv + control

        cost += time_discount * rectangle_size * (first_deriv ** 2 + lam * control**2)[0:basis_size].sum()
        mc = mc + time_increment * diff + torch.randn(1) * math.sqrt(time_increment) * noise

        mc[0] = 0
        mc[basis_size] = 0

    return cost

cost_diff_vec = vmap(finite_differences, in_dims=1, randomness="different")
def mc_average(input_point, mc_runs = 50000):
    eval_mat = input_point.unsqueeze(1).repeat(1, mc_runs)
    results = cost_diff_vec(eval_mat)
    return results.mean()

mc_average_vec = vmap(mc_average, in_dims = 0, randomness="different", chunk_size=10)
print('500 dim stochastic stationary dist cuda')
fd_matrix = torch.load("sample_points_stat_dist_500_dim_fd.pt").to(device="cuda")

mc_evals = mc_average_vec(fd_matrix[range(100),:])
torch.save(mc_evals, "mc_evals_stochastic_stat_dist_500_dim.pt")