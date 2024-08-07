import numpy as np
from .window_generic_model import WindowGenericModel


class SlidingWindow(WindowGenericModel):
    def __init__(self, input_chunk_length=None, output_chunk_length=None):
        super().__init__(
            input_chunk_length,
            output_chunk_length,
        )

    def embed_time_series(self, series):
        series = super()._embed_time_series(
            series,
        )
        if series == None:
            raise ValueError("This class is not defined")
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

        return data
