import numpy as np
import torch
import torch.nn as nn
import torch.func as func
from typing import Callable, Optional

#generic multi-layer feedforward neural network that does not have compact final weight support
class SimpleNN(nn.Module):
    def __init__(self, dim_in, num_layers, num_neurons, dim_out, bias = False):
        super(SimpleNN, self).__init__()
        self.dim_in = dim_in
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.dim_out = dim_out
        self.bias = bias #final bias

        layers = []

        # input layer
        layers.append(nn.Linear(self.dim_in, self.num_neurons))

        # hidden layers
        for _ in range(num_layers-1):
            layers.extend([nn.Sigmoid(), nn.Linear(self.num_neurons, self.num_neurons)])

        # output layer
        layers.extend([nn.Sigmoid(), nn.Linear(self.num_neurons, self.dim_out, bias = bias)])

        # build the network
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return self.network(x).squeeze()


class SimpleNNReLU(nn.Module):
    def __init__(self, dim_in, num_layers, num_neurons, dim_out, bias = False):
        super(SimpleNNReLU, self).__init__()
        self.dim_in = dim_in
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.dim_out = dim_out
        self.bias = bias #final bias

        layers = []

        # input layer
        layers.append(nn.Linear(self.dim_in, self.num_neurons))

        # hidden layers
        for _ in range(num_layers-1):
            layers.extend([nn.ReLU(), nn.Linear(self.num_neurons, self.num_neurons)])

        # output layer
        layers.extend([nn.ReLU(), nn.Linear(self.num_neurons, self.dim_out, bias=bias)])

        # build the network
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return self.network(x).squeeze()


#credit to GPT 5.2 for DGM code
class DGMBlock(nn.Module):
    """
    One DGM 'LSTM-like' gated layer.

    Given:
      X : (batch, in_dim)   original network input (e.g., (t,x) or x)
      S : (batch, hidden)   hidden state

    Computes gates Z,G,R and candidate H, then updates:
      S_new = (1 - G) * H + Z * S
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        act1: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        act2: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.act1 = act1
        self.act2 = act2

        # Each gate has an affine map of X and an affine map of S.
        # We include the bias on the X-map; the S-map uses bias=False
        # so the total bias is not duplicated.
        def lin_x() -> nn.Linear:
            return nn.Linear(in_dim, hidden_dim, bias=True)

        def lin_s() -> nn.Linear:
            return nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.Uz, self.Wz = lin_x(), lin_s()
        self.Ug, self.Wg = lin_x(), lin_s()
        self.Ur, self.Wr = lin_x(), lin_s()
        self.Uh, self.Wh = lin_x(), lin_s()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Xavier/Glorot initialization is conventional for DGM implementations.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, S: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        Z = self.act1(self.Uz(X) + self.Wz(S))
        G = self.act1(self.Ug(X) + self.Wg(S))
        R = self.act1(self.Ur(X) + self.Wr(S))
        H = self.act2(self.Uh(X) + self.Wh(S * R))
        S_new = (1.0 - G) * H + Z * S
        return S_new


class DGMNet(nn.Module):
    """
    DGMNet / Deep Galerkin Method network architecture.

    Use:
      - Elliptic PDE on R^n:        in_dim = n,    pass X = x
      - Parabolic PDE on [0,T]xR^n: in_dim = n+1,  pass X = (t,x)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        out_dim: int = 1,
        act_init: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        act1: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        act2: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        out_act: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.act_init = act_init
        self.out_act = out_act

        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [DGMBlock(in_dim, hidden_dim, act1=act1, act2=act2) for _ in range(n_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (batch, in_dim) typically, but may be (in_dim,) if user passes a single point.
        Returns:
          - (batch, out_dim) in general
          - a scalar (0-dim) tensor if the output has exactly one element (needed for torch.func.grad/hessian)
        """
        S = self.act_init(self.input_layer(X))
        for blk in self.blocks:
            S = blk(S, X)

        Y = self.output_layer(S)
        if self.out_act is not None:
            Y = self.out_act(Y)

        # CRITICAL: torch.func.grad / hessian require a scalar output.
        # Your f(z) calls qnet(z.unsqueeze(0)) -> shape (1,1). Convert to 0-dim scalar.
        if Y.numel() == 1:
            return Y.reshape(())  # true scalar tensor

        return Y