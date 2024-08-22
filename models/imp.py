import torch

from .torch_generic_model import TorchGenericModel

from .window.sliding_window import SlidingWindow


class MorphologicalLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(MorphologicalLayer, self).__init__()
        self.dilation_weight = torch.nn.Parameter(torch.randn(output_size, input_size))
        self.erosion_weight = torch.nn.Parameter(torch.randn(output_size, input_size))

    def forward(self, x):
        dilation = torch.max(x.unsqueeze(1) + self.dilation_weight, dim=2)[0]
        erosion = torch.min(x.unsqueeze(1) - self.erosion_weight, dim=2)[0]
        output = dilation + erosion
        return output


class IMP(TorchGenericModel):
    def __init__(
        self,
        input_chunk_length,
        output_chunk_length,
        batch_size,
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

        self.morph_layer = MorphologicalLayer(input_chunk_length, output_chunk_length)
        self.optimizer = optimizer_cls(self.parameters(), lr=lr)

    def forward(self, x):
        return self.morph_layer(x)

    def incremental_update(self, x, y, learning_rate=0.01):
        output = self.forward(x)
        loss = self.loss_fn(output, y)
        loss.backward()
        with torch.no_grad():
            for param in self.parameters():
                param -= learning_rate * param.grad
        self.zero_grad()

    def configure_optimizers(self):
        return self.optimizer


"""""
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
        out = torch.nn.functional.avg_pool1d(
            x, kernel_size=self.kernel_size, stride=1, padding=self.padding
        )
        out = self.embedding(out)
        out = self.morph_layer(out)
        out = self.output_layer(out)
        return out
""" ""
