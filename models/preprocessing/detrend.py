import numpy as np


class Detrend:
    def __init__(self):
        self.trend_line = None
        self.order = None

    def fit(self, data, order=1):
        # Garantir que data seja um array 2D
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        x = np.arange(0, len(data))
        self.trend_line = np.zeros_like(data)
        self.order = order

        # Ajustar para cada coluna em data
        for i in range(data.shape[1]):
            coeffs = np.polyfit(x, data[:, i], order)
            self.trend_line[:, i] = np.polyval(coeffs, x)

    def transform(self, data):
        # Garantir que data seja um array 2D
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        x = np.arange(0, len(self.trend_line))
        trend_line = np.zeros_like(data)

        # Ajustar para cada coluna em data
        for i in range(data.shape[1]):
            z = np.polyfit(x, self.trend_line[:, i], self.order)
            p = np.poly1d(z)
            if len(self.trend_line) != len(data):
                trend_line[:, i] = np.array(
                    [p(value + len(self.trend_line)) for value in range(len(data))]
                )
            else:
                trend_line[:, i] = self.trend_line[:, i]

        return data - trend_line

    def fit_transform(self, data, order=1):
        self.fit(data, order)
        return self.transform(data)

    def inverse_transform(self, data):
        # Garantir que data seja um array 2D
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        x = np.arange(0, len(self.trend_line))
        trend_line = np.zeros_like(data)

        # Ajustar para cada coluna em data
        for i in range(data.shape[1]):
            z = np.polyfit(x, self.trend_line[:, i], self.order)
            p = np.poly1d(z)
            if len(self.trend_line) != len(data):
                trend_line[:, i] = np.array(
                    [p(value + len(self.trend_line)) for value in range(len(data))]
                )
            else:
                trend_line[:, i] = self.trend_line[:, i]

        return data + trend_line
