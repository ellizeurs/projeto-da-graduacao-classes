
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
            if is_decreasing(self.last_losses):
                for i in range(1, self.patience):
                    if self.last_losses[i] > self.last_losses[i-1] + self.min_delta:
                        raise StopIteration('loss is increasing')
            else:
                raise StopIteration('loss is increasing')