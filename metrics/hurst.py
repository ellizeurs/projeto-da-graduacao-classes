import numpy as np
from ..TimeSeries import TimeSeries

import warnings

class Hurst:
    """
    H < 0,5: série temporal anti-persistente (traduz-se aproximadamente em mercado lateral).
    H = 0,5: passeio aleatório ou um mercado onde a previsão do futuro com base em dados passados ​​não é possível.
    H > 0,5: série temporal persistente (que se traduz aproximadamente em um mercado de tendência).
    """

    def __init__(self, q=2):
        self.q = q

    def calculate(self, S):
        if type(S) == TimeSeries:
            S = S.univariate_values()
        L = len(S)
        if L < 100:
            warnings.warn("Data series very short!")

        H = np.zeros((len(range(5, 20)), 1))
        k = 0

        for Tmax in range(5, 20):
            x = np.arange(1, Tmax + 1, 1)
            mcord = np.zeros((Tmax, 1))

            for tt in range(1, Tmax + 1):
                dV = S[np.arange(tt, L, tt)] - S[np.arange(tt, L, tt) - tt]
                VV = S[np.arange(tt, L + tt, tt) - tt]
                N = len(dV) + 1
                X = np.arange(1, N + 1, dtype=np.float64)
                Y = VV
                mx = np.sum(X) / N
                SSxx = np.sum(X**2) - N * mx**2
                my = np.sum(Y) / N
                SSxy = np.sum(np.multiply(X, Y)) - N * mx * my
                cc1 = SSxy / SSxx
                cc2 = my - cc1 * mx
                ddVd = dV - cc1
                VVVd = (
                    VV - np.multiply(cc1, np.arange(1, N + 1,
                                     dtype=np.float64)) - cc2
                )
                mcord[tt - 1] = np.mean(np.abs(ddVd) ** self.q) / np.mean(
                    np.abs(VVVd) ** self.q
                )

            mx = np.mean(np.log10(x))
            SSxx = np.sum(np.log10(x) ** 2) - Tmax * mx**2
            my = np.mean(np.log10(mcord))
            SSxy = (
                np.sum(np.multiply(np.log10(x), np.transpose(np.log10(mcord))))
                - Tmax * mx * my
            )
            H[k] = SSxy / SSxx
            k = k + 1

        mH = np.mean(H) / self.q

        return mH, "Random walk" if mH == 0.5 else (
            "Anti-persistente"
            if mH < 0.5
            else ("Persistente" if not np.isnan(mH) else "---")
        )
