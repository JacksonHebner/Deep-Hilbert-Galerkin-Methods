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

alpha = 1 #viscosity
basis_size = 500

def nonlin_operator(a):
    N = a.numel()
    x = a.view(1, 1, N)        # shape [batch=1, channels=1, length=N]

    # full self-convolution (length 2N-1)
    full_conv = torch.nn.functional.conv1d(x, x.flip(-1), padding=N-1)[0, 0]   # (2N-1,)
    # conv_k, 1 <= k <= N
    conv = a.new_zeros(N)
    conv[1:] = full_conv[:N-1]

    # full self-correlation (length 2N-1)
    full_corr = torch.nn.functional.conv1d(x, x, padding=N-1)[0, 0]            # (2N-1,)
    # autocorrelation at lags 0..N-1
    r = full_corr[N-1:N-1+N]                                 # r_0,...,r_{N-1}
    # corr_k, 1 <= k <= N-1, corr_N = 0
    corr = a.new_zeros(N)
    corr[:-1] = r[1:]                                        # r_1,...,r_{N-1}
    # corr[-1] already 0

    k = torch.arange(1, N+1)

    return (-alpha*k**2/4)* a + ((k/8) * conv - (k/4) * corr)/math.sqrt(math.pi)


num_points = 100
time_inc = 0.00001
time_inc_sqrt = math.sqrt(time_inc)
time_steps = 2000000

#for sigma = 1/sqrt(2pi)
k_int = torch.arange(1, basis_size+1)
odd = (k_int % 2 == 1)
const_proj = (4.0/torch.pi) * odd / k_int / math.sqrt(2)
adjuster = 1/(torch.arange(1,basis_size+1) ** 2)

def sample(dummy = 1):
    start = torch.randn(basis_size) * adjuster
    for i in range(time_steps):
        start += nonlin_operator(start) * time_inc + const_proj * torch.randn(1) * time_inc_sqrt
    return start

sample_vec = vmap(sample, randomness="different")
sample_points = sample_vec(torch.zeros(100))
torch.save(sample_points, "sample_points_stat_dist_500_dim.pt")