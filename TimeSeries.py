import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

from darts.models.filtering.moving_average_filter import MovingAverageFilter
from darts.utils.statistics import check_seasonality, plot_acf

def TimeSeriesRatio(self):
    ratio_series = self.pd_series().div(self.pd_series().shift(1))[1:]
    ratio_dataframe = ratio_series.to_frame(name=f"{self.columns[0]}_ratio")
    ratio_time_series = TimeSeries.from_dataframe(ratio_dataframe)
    return ratio_time_series


def TimeSeriesDetrend(self, order=1):
    z = np.polyfit(np.arange(0, len(self)), self.univariate_values(), order)
    p = np.poly1d(z)
    trend_line = TimeSeries.from_times_and_values(
        self.time_index, [p(value) for value in range(len(self))]
    )
    return self - trend_line


def TimeSeriesScalerFitAndInverseTransform(self, original):
    scaler = Scaler()
    scaler.fit(original)
    return scaler.inverse_transform(self)


def TimeSeriesInverseDetrend(self, original, order=1):
    z = np.polyfit(np.arange(0, len(original)), original.univariate_values(), order)
    p = np.poly1d(z)
    trend_line = TimeSeries.from_times_and_values(
        original.time_index, [p(value) for value in len(original)]
    )
    start_time = self.start_time()
    end_time = self.end_time()
    if self.time_index[-1].to_pydatetime() > trend_line.time_index[-1].to_pydatetime():
        append_dates = pd.date_range(
            original.end_time(), self.end_time(), freq=original.freq, name="Data"
        )[1:]
        trend_line = trend_line.concatenate(
            TimeSeries.from_times_and_values(
                append_dates, [p(value + len(original)) for value in len(append_dates)]
            )
        )
    return self + trend_line[start_time:end_time]


def TimeSeriesCheckSeasonality(self, alpha=0.05):
    for seasonality_period in range(2, len(self)):
        for max_lag in range(seasonality_period, len(self)):
            seasonality_exists, seasonality_period_l = check_seasonality(
                self, seasonality_period, max_lag, alpha
            )
            if seasonality_exists:
                return seasonality_period_l
    return False


def TimeSeriesPlotAcf(self):
    seasonality = self.check_seasonality()
    plot_acf(
        self,
        seasonality if seasonality else None,
        seasonality * 2 if seasonality * 2 > 24 else 24,
    )
    plt.suptitle(self.columns[0] + " - Autocorrelação")
    plt.title("Sazonalidade: {}".format(seasonality if seasonality else "-"))
    plt.show()


def TimeSeriesFilter(self, filter=MovingAverageFilter(10)):
    series_filtered = filter.filter(self)
    series_residuals = self - series_filtered
    return TimeSeries.from_dataframe(
        pd.DataFrame(
            {
                "Data": self.time_index,
                f"{self.columns[0]}_filtered": series_filtered.univariate_values(),
            }
        ),
        time_col="Data",
        value_cols=f"{self.columns[0]}_filtered",
    ), TimeSeries.from_dataframe(
        pd.DataFrame(
            {
                "Data": self.time_index,
                f"{self.columns[0]}_residuals": series_residuals.univariate_values(),
            }
        ),
        time_col="Data",
        value_cols=f"{self.columns[0]}_residuals",
    )


def TimeSeriesPlotFourierAnalisys(self):
    # Calcular a Transformada de Fourier
    fourier_transform = np.fft.fft(self.univariate_values())
    frequencies = np.fft.fftfreq(len(self))  # Frequências correspondentes

    # Calcular as amplitudes das frequências
    amplitudes = np.abs(fourier_transform)
    amplitudes = amplitudes[
        : len(amplitudes) // 2
    ]  # Considerar apenas as frequências positivas

    # Encontrar as frequências dominantes (com as maiores amplitudes)
    indices_dominantes = np.argsort(amplitudes)[::-1]
    frequencias_dominantes = frequencies[indices_dominantes]
    amplitudes_dominantes = amplitudes[indices_dominantes]

    ciclos = []
    for freq, amp in zip(frequencias_dominantes[:], amplitudes_dominantes[:]):
        if freq * len(self) > 1 and amp / sum(amplitudes_dominantes) > 0.1:
            ciclos.append("{:.4f}".format(freq))

    # Plotar a amplitude das frequências
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, np.abs(fourier_transform))
    plt.xlabel("Frequência")
    plt.ylabel("Amplitude")
    plt.suptitle(self.columns[0] + " - Análise de Fourier")
    plt.title("Ciclos: " + (", ".join(ciclos) if len(ciclos) > 0 else "-"))
    plt.show()


def TimeSeriesSplit(self, split):
    if type(split) == pd.Timestamp:
        return self.split_before(split)
    elif type(split) == list:
        series = []
        serie1, serie2 = self.split_before(split.pop(0))
        series.append(serie1)
        for end_time in split:
            serie1, serie2 = serie2.split_before(end_time)
            series.append(serie1)
        series.append(serie2)
        return tuple(series)
    else:
        raise TypeError("split is Union[pd.Timestamp or list[pd.Timestamp]]")


def TimeSeriesInverseRatio(self, start_value):
    actual_value = start_value.univariate_values()[0]
    pred_array = []
    for i, value_pred in enumerate(self):
        actual_value = value_pred * actual_value
        pred_array.append(actual_value.univariate_values())
    return TimeSeries.from_dataframe(
        pd.DataFrame({"Data": self.time_index, f"{self.columns[0]}": pred_array}),
        time_col="Data",
        value_cols=f"{self.columns[0]}",
        freq=self.freq,
    )


TimeSeries.ratio = TimeSeriesRatio
TimeSeries.detrend = TimeSeriesDetrend
TimeSeries.fit_inverse_transform = TimeSeriesScalerFitAndInverseTransform
TimeSeries.fit_inverse_detrend = TimeSeriesInverseDetrend
TimeSeries.check_seasonality = TimeSeriesCheckSeasonality
TimeSeries.plot_acf = TimeSeriesPlotAcf
TimeSeries.filter = TimeSeriesFilter
TimeSeries.plot_fourier_analisys = TimeSeriesPlotFourierAnalisys
TimeSeries.split = TimeSeriesSplit
TimeSeries.inverse_ratio = TimeSeriesInverseRatio