from torch_generic_model import TorchGenericModel

from window.sliding_window import SlidingWindow
from window.takens import Takens

import torch

class NARX(TorchGenericModel):
    def __init__(
        self,
        input_chunk_length,
        output_chunk_length,
        hidden_size=32,
        batch_size=32,
        lr=0.01,
        n_epochs=100,
        random_state=None,
        loss_fn=torch.nn.MSELoss(),
        optimizer_cls=torch.optim.Adam,
        window_model=SlidingWindow,
        device='cpu',
    ):
        super().__init__(
            input_chunk_length,
            output_chunk_length,
            batch_size,
            loss_fn,
            n_epochs,
            random_state,
            window_model,
            device,
        )

        self.hidden_size = hidden_size
        self.lin = torch.nn.Linear(self.input_chunk_length, self.hidden_size)
        self.lin2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.lin3 = torch.nn.Linear(self.hidden_size, self.output_chunk_length)
        self.tanh = torch.nn.Tanh()

        self.optimizer = optimizer_cls(self.parameters(), lr=lr)

    def forward(self, x):
        out = self.lin(x)
        out = self.tanh(out)
        out = self.lin2(out)
        out = self.tanh(out)
        out = self.lin3(out)
        return out
    

class NARXTakens(NARX):
    def __init__(
        self,
        input_chunk_length,
        output_chunk_length,
        hidden_size=32,
        batch_size=32,
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