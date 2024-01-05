
from ...functions import is_decreasing


class PatienceStopping:
    def __init__(self, patience = 10, min_delta = 1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.last_losses = []

    def __call__(self, loss):
        self.last_losses.insert(0, loss)
        self.last_losses = self.last_losses[:self.patience]

        if len(self.last_losses) == self.patience:
            if self.last_losses[0] > self.last_losses[-1] - self.min_delta:
                raise StopIteration('loss is increasing')
            
    def __str__(self):
        return f"{self.__class__.__name__}({', '.join(f'{nome}={valor}'
                                                for nome, valor
                                                in self.__dict__.items()
                                                if not nome.startswith("_"))})"