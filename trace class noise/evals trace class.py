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
model.load_state_dict(torch.load('No control, sigma = trace class, 25 func, 250 hilbert, 600 neur one layer, QPDE.pt2', weights_only=True, map_location=torch.device('cpu')))
model.eval()

#trace class covariance noise
sigma = 1/torch.arange(1,dim_cutoff_theoretical+1)

#target distribution is TCC

eigenvals = (torch.arange(1,dim_cutoff_theoretical+1) ** 2)/4

def true_eval(sigma, point, model):
    return (sigma**2 / (1 + 2*eigenvals)).sum() + (point**2 / (1 + 2*eigenvals)).sum()
def model_eval(sigma, point, model):
    point_proj = point[0:dim_in]
    return model(point_proj.unsqueeze(0))
def stationary_dist_sample(dummy = 1):
    return torch.randn(dim_cutoff_theoretical) * sigma / torch.sqrt(2*eigenvals)
def target_dist_sample(dummy = 1):
    return torch.randn(dim_cutoff_theoretical) * sigma / torch.sqrt(2*eigenvals)

true_eval_vec = func.vmap(true_eval, in_dims=(None, 0, None))
model_eval_vec = func.vmap(model_eval, in_dims=(None, 0, None))
stationary_dist_sample_vec = func.vmap(stationary_dist_sample, in_dims=(0), randomness="different")
target_dist_sample_vec = func.vmap(target_dist_sample, in_dims=(0), randomness="different")

times = 10000
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

#l^2(N) form of PDE operator -- plays well with vectorization
control = torch.zeros(dim_in)
lam, gamma, alpha = 1, 1, 1
doub = (torch.arange(1, dim_in+1))**2
A_vec = -alpha * doub / 4

trace_class_noise = sigma[0:dim_in]

def pde_op_combined(qnet, input_point):
    input_point_proj = input_point[0:dim_in]

    #original function, grad, and hessian with graph
    def f(z: torch.Tensor) -> torch.Tensor:
        return qnet(z.unsqueeze(0))
    g = func.grad(f)
    h = func.hessian(f)

    #concrete evaluations
    val = f(input_point_proj)                 # scalar
    val_grad = g(input_point_proj)            # shape (basis_size,)
    val_hess = h(input_point_proj)     # shape (basis_size, basis_size)

    in_prod = torch.dot(control, val_grad)
    diff = torch.sum(input_point ** 2)
    u_mag = lam * torch.sum(control ** 2)

    A_op = A_vec * input_point_proj
    drift = torch.dot(A_op, val_grad)

    trace_term = (1 / 2) *  ((trace_class_noise**2) * val_hess.diag()).sum()

    return -gamma*val + in_prod + diff + u_mag + drift + trace_term

pde_op_vec = func.vmap(pde_op_combined, in_dims=(None, 0))
pde_resids = pde_op_vec(model, points)
pde_residual_rmse = torch.sqrt((pde_resids**2).mean()).item()
print("PDE residual L^2 norm (RMSE):")
print(pde_residual_rmse)

def pde_deriv_errors(sigma, point, model):
    #original function, grad, and hessian with graph
    def f(z: torch.Tensor) -> torch.Tensor:
        input_point_proj = z[0:dim_in]
        return model(input_point_proj.unsqueeze(0)) - true_eval(sigma, z, model)
    g = func.grad(f)
    h = func.hessian(f)

    return g(point), h(point)

pde_deriv_errors_vec = func.vmap(pde_deriv_errors, in_dims=(None, 0, None))
errs = pde_deriv_errors_vec(sigma, points, model)

print("L^4 error:")
print(((errors**4).mean()**0.25).item())

first_deriv_errs = errs[0]
print("(w=4) H-norm of gradient error:")
print(((first_deriv_errs.norm(dim=1)**4).mean()**0.25).item())

second_deriv_errs = errs[1]
#print("(w=4) Operator-norm of Hessian error:")
#print(((torch.linalg.norm(second_deriv_errs, ord=2, dim=(1,2))**4).mean()**0.25).item())

#print("(w=4) Frobenius-norm of Hessian error:")
#print(((second_deriv_errs.norm(dim=(1,2))**4).mean()**0.25).item())

print("(w=4) Sobolev-type norm of Hessian error:")
points_2 = target_dist_sample_vec(torch.zeros(times))
sob = torch.matmul(second_deriv_errs, points_2.unsqueeze(-1)).squeeze(-1).norm(dim=1)
print(((sob**4).mean()**0.25).item())