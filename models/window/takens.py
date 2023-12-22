import numpy as np
from ...TimeSeries import TimeSeries

import torch

class Takens:
    def __init__(self, m=3, output=1, batch_size=4, tau=1):
        self.m = m
        self.tau = tau
        self.output = output
        self.batch_size = batch_size

    def embed_time_series(self, series):
        """
        Função para incorporar uma série temporal unidimensional em um espaço de fase de dimensão m.

        Parâmetros:
        - series: A série temporal unidimensional (numpy array).

        Retorna:
        - Uma matriz numpy com as sequências de m pontos incorporados no espaço de fase.
        """
        if type(series) == TimeSeries:
            series = series.univariate_values()
        N = len(series)
        embedded = np.zeros((N - (self.m - 1) * self.tau, self.m))

        for i in range(N - (self.m - 1) * self.tau):
            for j in range(self.m):
                embedded[i, j] = series[i + j * self.tau]

        inputs = embedded
        labels = []

        for input_l in inputs:
            labels.append(input_l[: -self.output])

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
        N, m = embedded_data.shape
        original_length = N + (m - 1) * self.tau
        original_data = np.zeros(original_length)

        for i in range(N):
            for j in range(m):
                original_data[i + j * self.tau] = embedded_data[i, j]

        return original_data