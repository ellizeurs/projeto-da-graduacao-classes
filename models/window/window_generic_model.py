from ...TimeSeries import TimeSeries


class WindowGenericModel:
    def __init__(self, input_chunk_length, output_chunk_length, batch_size):
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.batch_size = batch_size

    def embed_time_series(self, series):
        if (
            (self.input_chunk_length == None)
            or (self.output_chunk_length == None)
            or (self.batch_size == None)
        ):
            return None
        if type(series) == TimeSeries:
            series = series.univariate_values()
        return series

    def unembed_time_series(self, embedded_data):
        if (
            (self.input_chunk_length == None)
            or (self.output_chunk_length == None)
            or (self.batch_size == None)
        ):
            return None
        return embedded_data

    def __call__(self, input_chunk_length, output_chunk_length, batch_size):
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.batch_size = batch_size

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__, 
            ', '.join(f'{nome}={valor}'
                      for nome, valor
                      in self.__dict__.items()
                      if not nome.startswith("_")))
