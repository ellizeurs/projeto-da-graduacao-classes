from ...TimeSeries import TimeSeries


class WindowGenericModel:
    def __init__(self, input_chunk_length, output_chunk_length):
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

    def _embed_time_series(self, series):
        if (self.input_chunk_length == None) or (self.output_chunk_length == None):
            return None
        if type(series) == TimeSeries:
            series = series.univariate_values()
        return series

    def _unembed_time_series(self, input, target):
        if (self.input_chunk_length == None) or (self.output_chunk_length == None):
            return None
        return input, target

    def __call__(self, input_chunk_length, output_chunk_length):
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            ", ".join(
                f"{nome}={valor}"
                for nome, valor in self.__dict__.items()
                if not nome.startswith("_")
            ),
        )
