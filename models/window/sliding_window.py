import numpy as np
from ...TimeSeries import TimeSeries

import torch

class SlidingWindow:
    def __init__(self, input_chunk_length, output_chunk_length, batch_size):
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.batch_size = batch_size

    def embed_time_series(self, series):
        if type(series) == TimeSeries:
            series = series.univariate_values()
        data = []
        for i in range(
            0,
            len(series) - self.input_chunk_length - self.output_chunk_length + 1,
            self.output_chunk_length,
        ):
            input_sequence = series[i : i + self.input_chunk_length]
            target = series[
                i
                + self.input_chunk_length : i
                + self.input_chunk_length
                + self.output_chunk_length
            ]
            data.append((input_sequence, target))

        # Converter os dados em tensores
        inputs = torch.stack(
            [torch.from_numpy(np.array(item[0]).astype(np.float32)) for item in data]
        )
        labels = torch.stack(
            [torch.from_numpy(np.array(item[1]).astype(np.float32)) for item in data]
        )

        # Criar um DataLoader
        dataset = torch.utils.data.TensorDataset(inputs, labels)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )