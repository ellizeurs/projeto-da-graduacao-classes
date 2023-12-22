from TimeSeries import TimeSeries
import numpy as np
import pandas as pd

class Detrend:
    def __init__(self):
        self.trend_line = None
        self.order = None

    def fit(self, data, order=1):
        self.trend_line = data - data.detrend(order)
        self.order = order

    def transform(self, data):
        z = np.polyfit(
            np.arange(0, len(self.trend_line)),
            self.trend_line.univariate_values(),
            self.order,
        )
        p = np.poly1d(z)
        start_time = data.start_time()
        end_time = data.end_time()
        trend_line = self.trend_line
        if (
            data.time_index[-1].to_pydatetime()
            > trend_line.time_index[-1].to_pydatetime()
        ):
            append_dates = pd.date_range(
                trend_line.end_time(), data.end_time(), freq=self.trend_line.freq
            )[1:]
            trend_line = trend_line.concatenate(
                TimeSeries.from_dataframe(
                    pd.DataFrame(
                        {
                            "Data": append_dates,
                            f"{trend_line.columns[0]}": [
                                p(value + len(trend_line))
                                for value in range(len(append_dates))
                            ],
                        }
                    ),
                    time_col="Data",
                    value_cols=f"{trend_line.columns[0]}",
                    freq=self.trend_line.freq,
                )
            )
        trend_line = trend_line[start_time:end_time]
        return data - trend_line

    def fit_transform(self, data, order=1):
        self.fit(data, order)
        return self.transform(data)

    def inverse_transform(self, data):
        z = np.polyfit(
            np.arange(0, len(self.trend_line)),
            self.trend_line.univariate_values(),
            self.order,
        )
        p = np.poly1d(z)
        start_time = data.start_time()
        end_time = data.end_time()
        trend_line = self.trend_line
        if (
            data.time_index[-1].to_pydatetime()
            > trend_line.time_index[-1].to_pydatetime()
        ):
            append_dates = pd.date_range(
                trend_line.end_time(), data.end_time(), freq=self.trend_line.freq
            )[1:]
            trend_line = trend_line.concatenate(
                TimeSeries.from_dataframe(
                    pd.DataFrame(
                        {
                            "Data": append_dates,
                            f"{trend_line.columns[0]}": [
                                p(value + len(trend_line))
                                for value in range(len(append_dates))
                            ],
                        }
                    ),
                    time_col="Data",
                    value_cols=f"{trend_line.columns[0]}",
                    freq=self.trend_line.freq,
                )
            )
        trend_line = trend_line[start_time:end_time]
        return data + trend_line