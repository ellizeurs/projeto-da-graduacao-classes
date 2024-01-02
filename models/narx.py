import torch

from .torch_generic_model import TorchGenericModel

from .window.sliding_window import SlidingWindow


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
        window_model=SlidingWindow(),
        stopping_model=None,
        device="cpu",
    ):
        super().__init__(
            input_chunk_length,
            output_chunk_length,
            batch_size,
            loss_fn,
            n_epochs,
            random_state,
            window_model,
            stopping_model,
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
