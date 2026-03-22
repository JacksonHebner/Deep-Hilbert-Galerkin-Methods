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

alpha = 1 #this simplified problem is the 1D stochastic heat equation, so Af = \alpha f''
gamma  = 1
sigma = 1
lam = 1

#time_limit = 1 #i.e. T, the upper bound of time interval [0,T]
lower_bound = 0 #i.e. the L in [L,U]
upper_bound = 2 * torch.pi

#Basis is trigonometric in this code
#sin(x/2)/sqrt(pi), sin(x)/sqrt(pi), etc.
basis_size = 25 #i.e. d, for neural networks
sample_basis_size = 250 #i.e. N, for representing H

# Initialization
qnet = SimpleNN(dim_in=basis_size, num_neurons=600, num_layers=1, dim_out=1)

#Fixed control (could also be a function)
control = torch.zeros(basis_size)

#target function of the controlled pde, i.e. v_0
target_heat = torch.zeros(sample_basis_size)

doub = (torch.arange(1, basis_size+1))**2
A_vec = -alpha * doub / 4

#for sigma = trace class
k_int = torch.arange(1, sample_basis_size+1)
trace_class_noise_full = 1/k_int
trace_class_noise = trace_class_noise_full[0:basis_size]

#l^2(N) form of PDE operator -- plays well with vectorization
def pde_op_combined(qnet, input_point):
    input_point_proj = input_point[0:basis_size]

    #original function, grad, and hessian with graph
    def f(z: torch.Tensor) -> torch.Tensor:
        return qnet(z.unsqueeze(0))
    g = func.grad(f)
    #h = func.hessian(f)

    #concrete evaluations
    val = f(input_point_proj)                 # scalar
    val_grad = g(input_point_proj)            # shape (basis_size,)
    #val_hess = h(input_point_proj)     # shape (basis_size, basis_size)

    in_prod = torch.dot(control, val_grad)
    diff = torch.sum((target_heat - input_point) ** 2)
    u_mag = lam * torch.sum(control ** 2)

    A_op = A_vec * input_point_proj
    drift = torch.dot(A_op, val_grad)

    #trace_term = (sigma ** 2 / 2) *  ((trace_class_noise**2) * val_hess.diag()).sum()

    return -(-gamma*val + in_prod + diff + u_mag + drift).detach() * val

pde_op_vec = func.vmap(pde_op_combined, in_dims=(None, 0))

#learning rate
lr = 0.05
Qoptimizer = optim.Adam(qnet.parameters(), lr=lr)
Qscheduler = LambdaLR(Qoptimizer, lr_lambda=lambda epoch: (epoch + 20) ** -0.6)

Nmc = 2000 #number of points to sample in MC integration
epochs = 2000000 #number of training epochs

pde_training_losses, hams = [], []

#for determining reference measure (here stationary dist of white noise)
eigenvals = (torch.arange(1,sample_basis_size+1) ** 2)/4
adjuster = (sigma / (math.sqrt(2) * eigenvals)).repeat(Nmc, 1)

for j in range(0, epochs):
    if (j % 2 == 0):
        #random sampling of points in l^2 (NOTE: these functions have very large second derivatives!)
        grid = torch.randn(Nmc, sample_basis_size, requires_grad=True)
        grid = grid * adjuster

        pde_loss = pde_op_vec(qnet, grid).mean()
        pde_training_losses.append(pde_loss.item())

        Qoptimizer.zero_grad()
        pde_loss.backward()
        Qoptimizer.step()
        Qscheduler.step()

    if (j % 10 == 2):
        pde_test = qnet(torch.zeros(basis_size).unsqueeze(0)).item()
        print(j, pde_loss.item(), pde_test)

torch.save(qnet.state_dict(),
           "No control, no noise, 25 func, 250 hilbert, 600 neur one layer, WN stat, QPDE.pt2")


plt.style.use('ggplot')
plt.figure(figsize=(12,8))
plt.ylabel('Critic training loss')
plt.xlabel('Critic update epoch')
plt.yscale('log')
plt.plot(pde_training_losses)
plt.show()