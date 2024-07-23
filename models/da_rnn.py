from .torch_generic_model import TorchGenericModel

from .window.sliding_window import SlidingWindow
from .window.takens import Takens

import torch


class DA_RNN(TorchGenericModel):
    def __init__(
        self,
        input_chunk_length,
        output_chunk_length,
        hidden_size=32,
        batch_size=32,
        num_heads_attention=4,
        dropout=0.1,
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

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads_attention = num_heads_attention
        self.rnn = torch.nn.LSTM(self.input_chunk_length, self.hidden_size)
        self.attention = torch.nn.MultiheadAttention(
            self.hidden_size, self.num_heads_attention, self.dropout
        )
        self.fc = torch.nn.Linear(self.hidden_size, self.output_chunk_length)

        self.optimizer = optimizer_cls(self.parameters(), lr=lr)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out
