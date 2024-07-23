import pandas as pd
import numpy as np

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

import torch
import pytorch_lightning as pl

from ..TimeSeries import TimeSeries


class TorchGenericModel(pl.LightningModule):
    def __init__(
        self,
        input_chunk_length,
        output_chunk_length,
        batch_size,
        loss_fn,
        n_epochs,
        random_state,
        window_model,
        preprocessing,
        pl_trainer_kwargs,
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

        self.window_model = window_model
        self.preprocessing = preprocessing
        self.window_model(input_chunk_length, output_chunk_length)
        self.pl_trainer_kwargs = (
            pl_trainer_kwargs if pl_trainer_kwargs is not None else {}
        )

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        predictions = self.forward(inputs)
        loss = self.loss_fn(predictions, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        predictions = self.forward(inputs)
        loss = self.loss_fn(predictions, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer

    def fit(self, series, val_series=None):
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
            raise ValueError("Series must be Union(TimeSeries | list[TimeSeries])")

        for i, serie in enumerate(series):
            series[i] = self.apply_preprocessing(serie)

            inputs = torch.stack(
                [
                    torch.from_numpy(np.array(item[0]).astype(np.float32))
                    for item in series[i]
                ]
            )
            labels = torch.stack(
                [
                    torch.from_numpy(np.array(item[1]).astype(np.float32))
                    for item in series[i]
                ]
            )

            series[i] = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(inputs, labels),
                batch_size=self.batch_size,
                shuffle=False,
            )

        series = torch.utils.data.ConcatDataset([dl.dataset for dl in series])
        series = torch.utils.data.DataLoader(
            series, batch_size=self.batch_size, shuffle=False
        )

        if val_series:
            if type(val_series) == TimeSeries:
                self.last_index = val_series.time_index[-1]
                self.data_freq = val_series.freq
                val_series = val_series.univariate_values().tolist()
                self.last_data = val_series[-self.input_chunk_length :]
                val_series = [self.window_model.embed_time_series(val_series)]
            elif type(val_series) == list:
                val_series = [
                    self.window_model.embed_time_series(
                        serie.univariate_values().tolist()
                    )
                    for serie in val_series
                ]
            else:
                raise ValueError("Series must be Union(TimeSeries | list[TimeSeries])")

            for i, serie in enumerate(val_series):
                val_series[i] = self.apply_preprocessing(serie)

                inputs = torch.stack(
                    [
                        torch.from_numpy(np.array(item[0]).astype(np.float32))
                        for item in val_series[i]
                    ]
                )
                labels = torch.stack(
                    [
                        torch.from_numpy(np.array(item[1]).astype(np.float32))
                        for item in val_series[i]
                    ]
                )

                val_series[i] = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(inputs, labels),
                    batch_size=self.batch_size,
                    shuffle=False,
                )

            val_series = torch.utils.data.ConcatDataset(
                [dl.dataset for dl in val_series]
            )
            val_series = torch.utils.data.DataLoader(
                val_series, batch_size=self.batch_size, shuffle=False
            )

        self.fit_called = True

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        trainer = pl.Trainer(max_epochs=self.n_epochs, **self.pl_trainer_kwargs)
        trainer.fit(self, series, val_series)

    def predict_window(self, test_input):
        test_input_tensor = torch.tensor(test_input, dtype=torch.float32).view(
            1, self.input_chunk_length
        )
        output = self.forward(test_input_tensor)
        return output.cpu().detach().numpy().flatten()

    def apply_preprocessing(self, series):
        for preprocessing in self.preprocessing:
            for j, s in enumerate(series):
                series[j] = (
                    preprocessing.fit_transform(np.array(s[0]).reshape(-1, 1))
                    .flatten()
                    .tolist(),
                    preprocessing.transform(np.array(s[1]).reshape(-1, 1))
                    .flatten()
                    .tolist(),
                )
        return series

    def predict_window_with_preprocessing(self, test_input):
        test_input_preprocessed = self.apply_preprocessing([(test_input[0], [0])])
        output = self.predict_window(test_input_preprocessed[0][0])
        output = self.reverse_preprocessing(
            [(test_input_preprocessed[0][0], output)], [test_input]
        )
        return output[0][1]

    def reverse_preprocessing(self, series, original_series):
        for preprocessing in reversed(self.preprocessing):
            for j, (s, o_s) in enumerate(zip(series, original_series)):
                preprocessing.fit(np.array(o_s[0]).reshape(-1, 1))
                series[j] = (
                    preprocessing.inverse_transform(np.array(s[0]).reshape(-1, 1))
                    .flatten()
                    .tolist(),
                    preprocessing.inverse_transform(np.array(s[1]).reshape(-1, 1))
                    .flatten()
                    .tolist(),
                )
        return series

    def predict(self, n, series=None):
        if not self.fit_called:
            raise RuntimeError("fit() was not called before predict()")
        if series is None:
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

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        predicted_outputs = []

        with tqdm(
            total=n // self.output_chunk_length + 1, desc="Prediction"
        ) as progress_bar:
            for _ in range(n // self.output_chunk_length + 1):
                output = self.predict_window_with_preprocessing((test_input, [0]))
                for predicted_value in output:
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

    def evaluate(self, val):
        if not self.fit_called:
            raise RuntimeError("evaluate() was not called before predict()")
        if type(val) == TimeSeries:
            last_index = val.time_index[self.input_chunk_length]
            data_freq = val.freq
            val = val.univariate_values().tolist()
        else:
            raise ValueError("series must be TimeSeries")

        val = self.window_model.embed_time_series(val)
        val = self.apply_preprocessing(val)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        predicted_outputs = []

        with tqdm(total=len(val), desc="Prediction") as progress_bar:
            for val_l in val:
                output = self.predict_window_with_preprocessing(val_l[0])
                for predicted_value in output:
                    predicted_outputs.append(predicted_value)
                progress_bar.update(1)

        if type(data_freq) == int:
            predicted_outputs = TimeSeries.from_times_and_values(
                pd.Index(
                    [
                        i + data_freq + last_index
                        for i in range(0, data_freq * len(predicted_outputs), data_freq)
                    ]
                ),
                predicted_outputs,
            )
        else:
            predicted_outputs = TimeSeries.from_times_and_values(
                pd.date_range(
                    start=last_index, periods=len(predicted_outputs), freq=data_freq
                ),
                predicted_outputs,
            )
        return predicted_outputs

    def save_model(self, file_path):
        # Save model state and additional attributes on CPU
        self.cpu()  # Move the model to CPU before saving
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": (
                    self.optimizer.state_dict() if self.optimizer else None
                ),
                "attributes": {
                    k: v for k, v in self.__dict__.items() if not k.startswith("_")
                },
            },
            file_path,
        )

    def load_model(self, file_path):
        # Load model state and additional attributes
        checkpoint = torch.load(file_path, map_location="cpu")  # Load checkpoint on CPU

        # Load the state dict into the model
        self.load_state_dict(checkpoint["model_state_dict"])

        if (
            "optimizer_state_dict" in checkpoint
            and checkpoint["optimizer_state_dict"] is not None
        ):
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore additional attributes
        for k, v in checkpoint["attributes"].items():
            setattr(self, k, v)

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            ", ".join(
                f"{nome}={valor}"
                for nome, valor in self.__dict__.items()
                if not nome.startswith("_")
            ),
        )
