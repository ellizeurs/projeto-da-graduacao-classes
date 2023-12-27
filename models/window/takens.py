import numpy as np
from .window_generic_model import WindowGenericModel

import torch


class Takens(WindowGenericModel):
    def __init__(
        self,
        tau=1,
        input_chunk_length=None,
        output_chunk_length=None,
        batch_size=None,
    ):
        super().__init__(
            input_chunk_length,
            output_chunk_length,
            batch_size,
        )
        self.tau = tau

    def embed_time_series(self, series):
        """
        Função para incorporar uma série temporal unidimensional em um espaço de fase de dimensão m.

        Parâmetros:
        - series: A série temporal unidimensional (numpy array).

        Retorna:
        - Uma matriz numpy com as sequências de m pontos incorporados no espaço de fase.
        """
        series = super().embed_time_series(
            series,
        )
        if series == None:
            raise ValueError("This class is not defined")
        N = len(series)
        embedded = np.zeros((N - (self.input_chunk_length - 1) * self.tau, self.input_chunk_length))

        for i in range(N - (self.input_chunk_length - 1) * self.tau):
            for j in range(self.input_chunk_length):
                embedded[i, j] = series[i + j * self.tau]

        inputs = embedded
        labels = []

        for input_l in inputs:
            labels.append(input_l[: -self.output_chunk_length])

        # Converter os dados em tensores
        inputs = torch.stack(
            [torch.from_numpy(np.array(item).astype(np.float32)) for item in inputs]
        )
        labels = torch.stack(
            [torch.from_numpy(np.array(item).astype(np.float32)) for item in labels]
        )

        # Criar um DataLoader
        dataset = torch.utils.data.TensorDataset(inputs, labels)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )

    def unembed_time_series(self, embedded_data):
        """
        Função para reverter a incorporação do espaço de fase e obter a série temporal original.

        Parâmetros:
        - embedded_data: A matriz com as sequências de pontos incorporados no espaço de fase.

        Retorna:
        - A série temporal unidimensional reconstruída (numpy array).
        """
        embedded_data = super().embed_time_series(
            embedded_data,
        )
        if embedded_data == None:
            raise ValueError("This class is not defined")

        N, m = embedded_data.shape
        original_length = N + (m - 1) * self.tau
        original_data = np.zeros(original_length)

        for i in range(N):
            for j in range(m):
                original_data[i + j * self.tau] = embedded_data[i, j]

        return original_data
