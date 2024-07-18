import numpy as np
from .window_generic_model import WindowGenericModel


class Takens(WindowGenericModel):
    def __init__(
        self,
        tau=1,
        input_chunk_length=None,
        output_chunk_length=None,
    ):
        super().__init__(
            input_chunk_length,
            output_chunk_length,
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
        series = super()._embed_time_series(
            series,
        )
        if series == None:
            raise ValueError("This class is not defined")
        N = len(series)
        embedded = np.zeros(
            (
                N - (self.input_chunk_length + self.output_chunk_length - 1) * self.tau,
                self.input_chunk_length + self.output_chunk_length,
            )
        )

        for i in range(
            N - (self.input_chunk_length + self.output_chunk_length - 1) * self.tau
        ):
            for j in range(self.input_chunk_length + self.output_chunk_length):
                embedded[i, j] = series[i + j * self.tau]

        labels = []
        inputs = []

        data = []
        for i in embedded:
            data.append(
                (i[: -self.output_chunk_length], i[-self.output_chunk_length :])
            )

        return data
