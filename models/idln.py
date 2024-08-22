import torch

from .torch_generic_model import TorchGenericModel

from .window.sliding_window import SlidingWindow


class IDLN(TorchGenericModel):
    def __init__(
        self,
        input_chunk_length,
        output_chunk_length,
        batch_size,
        n_epochs=100,
        random_state=None,
        lr=1e-3,
        loss_fn=torch.nn.MSELoss(),
        optimizer_cls=torch.optim.Adam,
        window_model=SlidingWindow(),
        preprocessing=None,
        pl_trainer_kwargs=None,
    ):
        super().__init__(
            input_chunk_length,
            output_chunk_length,
            batch_size,
            loss_fn,
            n_epochs,
            random_state,
            window_model,
            preprocessing,
            pl_trainer_kwargs,
        )

        # Initializing IDLN-specific parameters
        self.a = torch.nn.Parameter(torch.randn(input_chunk_length))
        self.b = torch.nn.Parameter(torch.randn(input_chunk_length))
        self.c = torch.nn.Parameter(torch.randn(input_chunk_length))
        self.d = torch.nn.Parameter(torch.randn(input_chunk_length))
        self.p = torch.nn.Parameter(torch.randn(input_chunk_length))

        self.phi = torch.nn.Parameter(torch.rand(1))
        self.epsilon = torch.nn.Parameter(torch.rand(1))
        self.omega = torch.nn.Parameter(torch.rand(1))
        self.theta = torch.nn.Parameter(torch.rand(1))
        self.kappa = torch.nn.Parameter(torch.rand(1))
        self.alpha = torch.nn.Parameter(torch.rand(1))
        self.lambda_ = torch.nn.Parameter(torch.rand(1))

        self.optimizer = optimizer_cls(self.parameters(), lr=lr)

    def dilation(self, x):
        return torch.max(x + self.a, dim=-1)[0]

    def erosion(self, x):
        return torch.min(x + self.b, dim=-1)[0]

    def anti_dilation(self, x):
        return torch.min(x + self.c, dim=-1)[0]

    def anti_erosion(self, x):
        return torch.max(x + self.d, dim=-1)[0]

    def forward(self, x):
        # Applying morphological operations
        delta = self.dilation(x)
        epsilon = self.erosion(x)
        delta_bar = self.anti_dilation(x)
        epsilon_bar = self.anti_erosion(x)
        beta = torch.sum(x * self.p, dim=-1)

        # Linear combinations with weights φ, ε, ω, θ, κ, α, and λ
        term1 = self.phi * delta + (1 - self.phi) * epsilon
        term2 = self.omega * delta_bar + (1 - self.omega) * epsilon_bar
        term3 = self.lambda_ * term1 + (1 - self.lambda_) * beta
        output = self.alpha * term2 + (1 - self.alpha) * term3

        return output


"""""
class IDLN(TorchGenericModel):
    def __init__(
        self,
        input_chunk_length,
        output_chunk_length,
        bias=False,
        batch_size=32,
        lr=1e-3,
        n_epochs=100,
        random_state=None,
        loss_fn=torch.nn.MSELoss(),
        optimizer_cls=torch.optim.Adam,
        window_model=SlidingWindow(),
        preprocessing=None,
        pl_trainer_kwargs=None,
    ):
        super().__init__(
            input_chunk_length,
            output_chunk_length,
            batch_size,
            loss_fn,
            n_epochs,
            random_state,
            window_model,
            preprocessing,
            pl_trainer_kwargs,
        )

        self.bias = bias
        self.linear = torch.nn.Linear(input_chunk_length, output_chunk_length, bias)

        self.optimizer = optimizer_cls(self.parameters(), lr=lr)

    def forward(self, x):
        out = self.linear(x)
        return out
""" ""
