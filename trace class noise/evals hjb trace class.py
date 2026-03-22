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
model.load_state_dict(torch.load("Critic, trained control, sigma = trace class, 25 func, 250 hilbert, 600 neur one layer, A2.pt2", weights_only=True, map_location=torch.device('cpu')))
model.eval()
model_actor = SimpleNN(dim_in=dim_in, num_neurons=600, num_layers=1, dim_out=dim_in)
model_actor.load_state_dict(torch.load("Actor, trained control, sigma = trace class, 25 func, 250 hilbert, 600 neur one layer, A2.pt2", weights_only=True, map_location=torch.device('cpu')))
model_actor.eval()

#trace class covariance noise
sigma = 1 / torch.arange(1, dim_cutoff_theoretical + 1)

#target measure is TCC

eigenvals = (torch.arange(1,dim_cutoff_theoretical+1) ** 2)/4

M = 2 / (2 * eigenvals + 1 + torch.sqrt((2 * eigenvals + 1) ** 2 + 4))
Q = 0
R = M * sigma ** 2

def true_eval_critic(sigma, point, model):
    true_values = M * point ** 2 + Q * point + R
    return true_values.sum()
def model_eval_critic(sigma, point, model):
    point_proj = point[0:dim_in]
    return model(point_proj.unsqueeze(0))
def true_eval_actor(sigma, point, model_actor):
    true_values = -M * point - Q/2
    return true_values
def model_eval_actor(sigma, point, model_actor):
    point_proj = point[0:dim_in]
    out_first = model_actor(point_proj.unsqueeze(0)).squeeze(0)  # (dim_in,)
    pad = point.new_zeros(dim_cutoff_theoretical - dim_in)
    return torch.cat([out_first, pad], dim=0)

true_eval_critic_vec = func.vmap(true_eval_critic, in_dims=(None, 0, None))
model_eval_critic_vec = func.vmap(model_eval_critic, in_dims=(None, 0, None))
true_eval_actor_vec = func.vmap(true_eval_actor, in_dims=(None, 0, None))
model_eval_actor_vec = func.vmap(model_eval_actor, in_dims=(None, 0, None))

def stationary_dist_sample(dummy = 1):
    return torch.randn(dim_cutoff_theoretical) * sigma / torch.sqrt(2*eigenvals)
def target_dist_sample(dummy = 1):
    return torch.randn(dim_cutoff_theoretical) * sigma / torch.sqrt(2*eigenvals)

stationary_dist_sample_vec = func.vmap(stationary_dist_sample, in_dims=(0), randomness="different")
target_dist_sample_vec = func.vmap(target_dist_sample, in_dims=(0), randomness="different")

times = 1000000

points = target_dist_sample_vec(torch.zeros(times))
critic_model_vals = model_eval_critic_vec(sigma, points, model)
critic_true_vals = true_eval_critic_vec(sigma, points, model)
critic_errors = critic_model_vals - critic_true_vals

actor_model_vals = model_eval_actor_vec(sigma, points, model_actor)
actor_true_vals = true_eval_actor_vec(sigma, points, model_actor)
actor_errors = actor_model_vals - actor_true_vals

print("Critic L^1 error (ME):")
print(torch.abs(critic_errors).mean().item())
print("Critic L^2 error (RMSE):")
print((torch.sqrt((critic_errors**2).mean())).item())
print("Critic relative error 1:")
print(((torch.abs(critic_errors) / torch.abs(critic_true_vals)).mean()).item())
print("Critic relative error 2:")
print(torch.sqrt((critic_errors**2).sum() / (critic_true_vals**2).sum()).item())

print("Actor L^1 error (ME):")
print(actor_errors.norm(p=2, dim=1).mean().item())
print("Actor L^2 error (RMSE):")
print((torch.sqrt((dim_cutoff_theoretical * actor_errors**2).mean())).item())
print("Actor relative error 1:")
print((actor_errors.norm(p=2, dim = 1) / actor_true_vals.norm(p=2, dim = 1)).mean().item())
print("Actor relative error 2:")
print(torch.sqrt((actor_errors**2).sum() / (actor_true_vals**2).sum()).item())

# def pde_deriv_errors(sigma, point, model):
#     #original function, grad, and hessian with graph
#     def f(z: torch.Tensor) -> torch.Tensor:
#         input_point_proj = z[0:dim_in]
#         return model(input_point_proj.unsqueeze(0)) - true_eval_critic(sigma, z, model)
#     g = func.grad(f)
#     h = func.hessian(f)
#
#     return g(point), h(point)
#
# pde_deriv_errors_vec = func.vmap(pde_deriv_errors, in_dims=(None, 0, None))
# errs = pde_deriv_errors_vec(sigma, points, model)
#
# print("L^4 error:")
# print(((critic_errors**4).mean()**0.25).item())
#
# first_deriv_errs = errs[0]
# print("(w=4) H-norm of gradient error:")
# print(((first_deriv_errs.norm(dim=1)**4).mean()**0.25).item())
#
# second_deriv_errs = errs[1]
# print("(w=4) Operator-norm of Hessian error:")
# print(((torch.linalg.norm(second_deriv_errs, ord=2, dim=(1,2))**4).mean()**0.25).item())
#
# print("(w=4) Frobenius-norm of Hessian error:")
# print(((second_deriv_errs.norm(dim=(1,2))**4).mean()**0.25).item())
#
# print("(w=4) Sobolev-type norm of Hessian error:")
# points_2 = target_dist_sample(torch.zeros(times))
# sob = torch.matmul(second_deriv_errs, points_2).norm(dim=1)
# print(((sob**4).mean()**0.25).item())