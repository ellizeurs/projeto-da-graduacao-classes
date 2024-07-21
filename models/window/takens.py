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

        inputs = []
        targets = []

        # Itera sobre a série temporal para criar os inputs e targets
        for i in range(
            N - self.input_chunk_length * self.tau - self.output_chunk_length + 1
        ):
            try:
                target_vector = [
                    data[i + j * self.tau + self.input_chunk_length * self.tau]
                    for j in range(self.output_chunk_length)
                ]
            except:
                target_vector = [None for j in range(self.output_chunk_length)]
            targets.append(target_vector)

            # Cria um vetor de input
            input_vector = [data[i + j * self.tau] for j in self.input_chunk_length]
            inputs.append(input_vector)

        data = []
        for input, target in zip(inputs, targets):
            data.append((input.copy(), target.copy()))

        return data

    def unembed_time_series(self, input, target):
        """
        Reverte o processo de criação de embedding e alvos para séries temporais.

        Parameters:
        - inputs: Matriz de inputs para o modelo.
        - targets: Matriz de targets para o modelo.

        Returns:
        - data_reconstructed: Série temporal reconstruída.
        """

        series = super()._unembed_time_series(input, target)
        if series == None:
            raise ValueError("This class is not defined")
        N = len(series)

        inputs_aux, targets_aux = self.embed_time_series([i for i in range(10000)])

        data_reconstructed = [None]

        while data_reconstructed[-1] != None or len(data_reconstructed) == 1:
            for i in range(10000):
                data_reconstructed.append(None)
            for input_i, input in zip(inputs_aux, series):
                for input_j, value in zip(input_i, input[0]):
                    if value != None:
                        try:
                            data_reconstructed[input_j] = value
                        except:
                            pass

            for target_i, target in zip(targets_aux, series):
                for target_j, value in zip(target_i, target[1]):
                    if value != None:
                        try:
                            data_reconstructed[target_j] = value
                        except:
                            pass

            for i in reversed(range(len(data_reconstructed))):
                if data_reconstructed[i] == None:
                    data_reconstructed = data_reconstructed[:i]
                else:
                    break

        return data_reconstructed
