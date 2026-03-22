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

torch.set_default_device('cuda')

alpha = 1 #this simplified problem is the 1D stochastic Burger's equation, so Af = \alpha f'' + ff'
gamma  = 1
sigma = 1
lam = 1

#Basis is trigonometric: sin(x/2)/sqrt(pi), sin(x)/sqrt(pi), etc.
#PDE domain is [0,2pi] with Dirichlet conditions x(0) = x(2pi) = 0
basis_size = 100
sample_basis_size = 500 #i.e. N, for representing H

# Initialization
qnet = DGMNet(in_dim = basis_size, n_layers = 3)

#for sigma = 1/sqrt(2pi)
k_int = torch.arange(1, basis_size+1)
odd = (k_int % 2 == 1)
const_proj = (4.0/torch.pi) * odd / k_int / math.sqrt(2)

#returns Burgers operator \alpha*f'' + f*f'
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

#l^2(N) form of PDE operator -- plays well with vectorization
def pde_op_combined(qnet, input_point):
    con = torch.zeros(basis_size)
    input_point_proj = input_point[0:basis_size]

    #original function, grad, and hessian with graph
    def f(z: torch.Tensor) -> torch.Tensor:
        return qnet(z.unsqueeze(0))
    g = func.grad(f)
    h = func.hessian(f)

    #concrete evaluations
    val = f(input_point_proj)                 # scalar
    val_grad = g(input_point_proj)            # shape (basis_size,)
    val_hess = h(input_point_proj)     # shape (basis_size, basis_size)

    in_prod = torch.dot(con, val_grad)

    first_deriv = torch.arange(1, sample_basis_size+1)/2 #in cosines, but still orthonormal
    deriv_mag = torch.sum((first_deriv * input_point) ** 2)

    u_mag = lam * torch.sum(con ** 2)

    A_op = nonlin_operator(input_point)[0:basis_size]
    drift = torch.dot(A_op, val_grad)

    trace_term = (sigma ** 2 / 2) *  (torch.outer(const_proj, const_proj) * val_hess).sum()

    return -(-gamma*val + in_prod + deriv_mag + u_mag + drift + trace_term).detach() * val

pde_op_vec = func.vmap(pde_op_combined, in_dims=(None, 0))

#learning rate
lr = 0.05
Qoptimizer = optim.Adam(qnet.parameters(), lr=lr)
Qscheduler = LambdaLR(Qoptimizer, lr_lambda=lambda epoch: (epoch + 20) ** -0.6)

Nmc = 2000 #number of points to sample in MC integration
epochs = 2000000 #number of training epochs

pde_training_losses = []

test = torch.zeros(basis_size)

adjuster = 1/(torch.arange(1,sample_basis_size+1) ** 2)

for j in range(0, epochs):
    if (j % 2 == 0):
        grid = torch.randn(Nmc, sample_basis_size, requires_grad=True) * adjuster

        pde_loss = pde_op_vec(qnet, grid).mean()
        pde_training_losses.append(pde_loss.item())

        Qoptimizer.zero_grad()
        pde_loss.backward()
        Qoptimizer.step()
        Qscheduler.step()

    if (j % 10 == 2):
        pde_test = qnet(test.unsqueeze(0)).item()
        print(j, pde_loss.item(), pde_test)

torch.save(qnet.state_dict(), "sigma = 1sqrt(2pi), 100 func, 500 hilbert, DGMNet, TCC tail, QPDE fixed.pt2")