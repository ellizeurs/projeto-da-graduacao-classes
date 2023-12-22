import pandas as pd
import numpy as np

from tqdm import tqdm

import torch

from ..TimeSeries import TimeSeries

class TorchGenericModel(torch.nn.Module):
    def __init__(
        self,
        input_chunk_length,
        output_chunk_length,
        batch_size,
        loss_fn,
        n_epochs,
        random_state,
        window_model,
        device,
    ):
        super().__init__()
        self.n_epochs = n_epochs
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.batch_size = batch_size
        self.random_state = random_state

        self.fit_called = False

        self.loss_fn = loss_fn
        self.optimizer = None

        self.window_model = window_model(
            input_chunk_length, output_chunk_length, batch_size
        )
        self.device = device

    def forward(self, x):
        return x

    def fit(self, series):
        if type(series) == TimeSeries:
            self.last_index = series.time_index[-1]
            self.data_freq = series.freq
            series = series.univariate_values().tolist()
            self.last_data = series[-self.input_chunk_length :]
            series = [self.window_model.embed_time_series(series)]
        elif type(series) == list:
            series = [
                self.window_model.embed_time_series(serie.univariate_values().tolist())
                for serie in series
            ]
        else:
            raise ValueError("series must be Union(TimeSeries | list[TimeSeries])")
        self.fit_called = True

        if self.random_state != None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        model = self.to(self.device)

        try:
            with tqdm(total=self.n_epochs, desc="Epoch") as progress_bar:
                for _ in range(self.n_epochs):
                    total_loss = 0.0
                    for serie in series:
                        for batch_inputs, batch_labels in serie:
                            predictions = model.forward(batch_inputs.to(self.device))
                            loss = self.loss_fn(predictions, batch_labels.to(self.device))

                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            total_loss += loss.item()

                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=total_loss)
        except KeyboardInterrupt:
            pass

    def predict(self, n, series=None):
        if not self.fit_called:
            raise RuntimeError("fit() was not called before predict()")
        if series == None:
            try:
                test_input = self.last_data
                data_freq = self.data_freq
                last_index = self.last_index
            except:
                raise RuntimeError(
                    "series must be provided, for training with multiple time series"
                )
        elif type(series) == TimeSeries:
            last_index = series.time_index[-1]
            data_freq = series.freq
            series = series.univariate_values().tolist()
            test_input = series[-self.input_chunk_length :]
        else:
            raise ValueError("series must be TimeSeries")

        if self.random_state != None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        predicted_outputs = []
        model = self.to(self.device)
        with tqdm(total=n, desc="Prediction") as progress_bar:
            for _ in range(n // self.output_chunk_length + 1):
                output = model.forward(
                    torch.tensor(test_input, dtype=torch.float32).view(
                        1, self.input_chunk_length
                    ).to(self.device)
                )
                for predicted_value in output.numpy(force=True)[0]:
                    predicted_outputs.append(predicted_value)
                    test_input.append(predicted_value)
                test_input = test_input[self.output_chunk_length :]
                progress_bar.update(1)
        if type(data_freq) == int:
            predicted_outputs = TimeSeries.from_times_and_values(
                pd.Index(
                    [
                        i + data_freq + last_index
                        for i in range(0, data_freq * n, data_freq)
                    ]
                ),
                predicted_outputs[:n],
            )
        else:
            predicted_outputs = TimeSeries.from_times_and_values(
                pd.date_range(start=last_index, periods=n + 1, freq=data_freq)[1:],
                predicted_outputs[:n],
            )
        return predicted_outputs