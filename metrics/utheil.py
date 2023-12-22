import numpy as np

class UTheil:
    def __init__(self):
        pass

    def calculateU1(self, y_true, y_pred):
        """
        Calcula a métrica U1 de Theil.

        A métrica U1 avalia a precisão das previsões em relação à variabilidade dos dados observados.
        Ela é normalizada no intervalo [0, 1], onde 0 é o melhor desempenho e 1 é o pior.

        Parâmetros:
        - y_true: Valores reais.
        - y_pred: Valores previstos pelo modelo.

        Retorna:
        - Theil U1, no intervalo [0, 1].
        """
        y_true = y_true.univariate_values()
        y_pred = y_pred.univariate_values()
        N = len(y_true)

        # Numerador: Raiz quadrada da média dos quadrados das diferenças entre valores reais e previstos.
        numerator_squared = np.square(y_true - y_pred)
        mean_numerator = np.mean(numerator_squared)
        numerator = np.sqrt(mean_numerator)

        # Denominador: Raiz quadrada da média dos quadrados dos valores reais e dos valores previstos.
        true_squared = np.square(y_true)
        pred_squared = np.square(y_pred)
        mean_true = np.mean(true_squared)
        mean_pred = np.mean(pred_squared)
        denominator = np.sqrt(mean_true + mean_pred)

        # Calcula Theil U1.
        theilU1 = numerator / denominator

        return theilU1

    def calculateU2(self, y_true, y_pred):
        """
        Calcula a métrica U2 de Theil.

        A métrica U2 compara o desempenho do modelo com um modelo ingênuo.
        U2 < 1 indica que o modelo é superior ao ingênuo, U2 = 1 indica equivalência,
        e U2 > 1 indica que o modelo é inferior ao ingênuo.

        Parâmetros:
        - y_true: Valores reais.
        - y_pred: Valores previstos pelo modelo.

        Retorna:
        - Theil U2.
        """
        y_true = y_true.univariate_values()
        y_pred = y_pred.univariate_values()
        N = len(y_true)

        errors_relative = (y_pred[1:] - y_true[1:]) / y_true[:-1]
        errors_relative_squared = np.square(errors_relative)

        numerator = np.sqrt(np.sum(errors_relative_squared))

        true_relative = (y_true[1:] - y_true[:-1]) / y_true[:-1]
        true_relative_squared = np.square(true_relative)

        denominator = np.sqrt(np.sum(true_relative_squared))

        theilU2 = numerator / denominator

        return theilU2