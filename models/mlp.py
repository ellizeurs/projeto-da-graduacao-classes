import torch

from .torch_generic_model import TorchGenericModel

from .window.sliding_window import SlidingWindow


class MLP(TorchGenericModel):
    def __init__(
        self,
        input_chunk_length,
        output_chunk_length,
        hidden_size=32,
        batch_size=32,
        lr=1e-3,
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
        self.fc1 = torch.nn.Linear(input_chunk_length, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_chunk_length)
        self.tanh = torch.nn.Tanh()

        self.optimizer = optimizer_cls(self.parameters(), lr=lr)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
