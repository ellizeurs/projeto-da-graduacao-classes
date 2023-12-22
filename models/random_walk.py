import pandas as pd

import torch

from ..TimeSeries import TimeSeries

import random

class RandomWalkModel:
    def __init__(self, step=1, random_state=None):
        self.step = step
        self.random_state = random_state
        self.fitted = False

    def fit(self, series):
        self.last_value = series.univariate_values()[-1]
        self.last_date = series.end_time()
        self.first_value = series.univariate_values()[0]
        self.first_date = series.start_time()
        self.column_name = series.columns[0]
        self.fitted = True

    def predict(self, n, series=None):
        if not self.fitted:
            raise RuntimeError("This model not fitted")
        if self.random_state:
            random.seed(self.random_state)
        predictions = list()
        history = self.last_value
        for i in range(n):
            yhat = history + (-self.step if random.random() < 0.5 else self.step)
            predictions.append(yhat)
            history = yhat
        return TimeSeries.from_dataframe(
            pd.DataFrame(
                {
                    "Data": pd.date_range(
                        self.last_date, periods=n + 1, freq=self.last_date.freq
                    )[1:],
                    self.column_name: predictions,
                }
            ),
            time_col="Data",
            value_cols=self.column_name,
        )

    def historical_forecasts(self, series=None):
        if not self.fitted:
            raise RuntimeError("This model not fitted")
        if self.random_state:
            random.seed(self.random_state)
        predictions = list()
        dates = pd.date_range(self.first_date, self.last_date, freq=self.last_date.freq)
        history = self.first_value
        predictions.append(history)
        for _ in dates:
            yhat = history + (-self.step if random.random() < 0.5 else self.step)
            predictions.append(yhat)
            history = yhat
        return TimeSeries.from_dataframe(
            pd.DataFrame({"Data": dates, self.column_name: predictions[:-1]}),
            time_col="Data",
            value_cols=self.column_name,
        )

    def __repr__(self):
        return "RandomWalkModel(step = {})".format(self.step)