import torch
import torch.nn as nn
from .torch_generic_model import TorchGenericModel

from .window.sliding_window import SlidingWindow


class NLinear(TorchGenericModel):
    def __init__(
        self,
        input_chunk_length,
        output_chunk_length,
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

        # Linear layer that will transform input_chunk_length into output_chunk_length
        self.linear = nn.Linear(input_chunk_length, output_chunk_length)

        self.optimizer = optimizer_cls(self.parameters(), lr=lr)

    def forward(self, x):
        # Flatten the input tensor, apply the linear transformation, and reshape it back
        x = x.view(x.size(0), -1)  # Flatten the input
        output = self.linear(x)
        return output
