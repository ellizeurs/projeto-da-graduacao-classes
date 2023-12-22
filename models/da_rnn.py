from torch_generic_model import TorchGenericModel

from window.sliding_window import SlidingWindow
from window.takens import Takens

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
        lr=0.01,
        n_epochs=100,
        random_state=None,
        loss_fn=torch.nn.MSELoss(),
        optimizer_cls=torch.optim.Adam,
        window_model=SlidingWindow,
    ):
        super().__init__(
            input_chunk_length,
            output_chunk_length,
            batch_size,
            loss_fn,
            n_epochs,
            random_state,
            window_model,
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
    

class DA_RNN_Takens(DA_RNN):
    def __init__(
        self,
        input_chunk_length,
        output_chunk_length,
        hidden_size=32,
        batch_size=32,
        num_heads_attention=4,
        dropout=0.1,
        lr=0.01,
        n_epochs=100,
        random_state=None,
        takens_kwargs = {'tau': 1},
        loss_fn=torch.nn.MSELoss(),
        optimizer_cls=torch.optim.Adam,
    ):
        super().__init__(
            input_chunk_length,
            output_chunk_length,
            hidden_size,
            batch_size,
            num_heads_attention,
            dropout,
            lr,
            n_epochs,
            random_state,
            loss_fn,
            optimizer_cls,
        )

        self.takens_tau = takens_kwargs['tau']
        self.window_model = Takens(
            input_chunk_length, output_chunk_length, batch_size, takens_kwargs['tau']
        )

        self.optimizer = optimizer_cls(self.parameters(), lr=lr)