import torch
import torch.nn as nn
from .torch_generic_model import TorchGenericModel

from .window.sliding_window import SlidingWindow


class NHiTSBlock(nn.Module):
    def __init__(self, input_size, output_size, num_layers, layer_size):
        super(NHiTSBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NHiTS(TorchGenericModel):
    def __init__(
        self,
        input_chunk_length,
        output_chunk_length,
        num_stacks=3,
        num_blocks=1,
        num_layers=2,
        layer_size=128,
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

        self.blocks = nn.ModuleList(
            [
                NHiTSBlock(
                    input_chunk_length, output_chunk_length, num_layers, layer_size
                )
                for _ in range(num_stacks * num_blocks)
            ]
        )

        self.optimizer = optimizer_cls(self.parameters(), lr=lr)

    def forward(self, x):
        block_outputs = [block(x) for block in self.blocks]
        return sum(block_outputs)
