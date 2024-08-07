import torch

from .torch_generic_model import TorchGenericModel

from .window.sliding_window import SlidingWindow


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
