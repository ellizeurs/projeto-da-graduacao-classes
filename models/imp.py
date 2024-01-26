import torch

from .torch_generic_model import TorchGenericModel

from .window.sliding_window import SlidingWindow


class IMP(TorchGenericModel):
    def __init__(
        self,
        input_chunk_length,
        output_chunk_length,
        hidden_size=32,
        batch_size=32,
        kernel_size=3,
        padding=1,
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
        self.kernel_size = kernel_size
        self.padding = padding

        self.embedding = torch.nn.Linear(input_chunk_length, hidden_size)
        self.morph_layer = torch.nn.Sequential(
        #    torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
            torch.nn.ReLU(),
        #    torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
        )
        self.output_layer = torch.nn.Linear(hidden_size, output_chunk_length)

        self.optimizer = optimizer_cls(self.parameters(), lr=lr)

    def forward(self, x):
        out = torch.nn.functional.avg_pool1d(x, kernel_size=self.kernel_size, stride=1, padding=self.padding//2)
        out = self.embedding(out)
        out = self.morph_layer(out)
        out = self.output_layer(out)
        return out
