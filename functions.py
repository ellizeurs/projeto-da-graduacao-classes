import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta
import seaborn as sns

import torch

from .TimeSeries import TimeSeries
from darts.metrics import mape

from .const import FIG_SIZE
from .metrics import sle


def plot_trend_line(series, title=None, order=1):
    plt.figure(figsize=FIG_SIZE)
    if title:
        plt.title(title)
    series.plot()
    (series - series.detrend(order)).plot(label="trend line")
    plt.show()


def plot_series(series, title=None):
    plt.figure(figsize=FIG_SIZE)
    if title:
        plt.title(title)
    if type(series) == TimeSeries:
        series.plot()
    elif type(series) == list:
        for serie in series:
            serie.plot()
    plt.show()


def plot_series_labels(series, labels, title=None):
    plt.figure(figsize=FIG_SIZE)
    if type(series) == TimeSeries:
        series.plot()
    elif type(series) == list:
        for serie, label in zip(series, labels):
            serie.plot(label=label)
    if title:
        plt.title(title)
    plt.show()


def plot_correlation_matrix(series, annot=False, **kwargs):
    # Calculando a matriz de correlação
    correlation_matrix = np.corrcoef([serie.univariate_values() for serie in series])

    # Gerando o heatmap
    plt.figure(figsize=FIG_SIZE)
    ax = sns.heatmap(
        correlation_matrix,
        annot=annot,
        cmap="coolwarm",
        xticklabels=[serie.columns[0] for serie in series],
        yticklabels=[serie.columns[0] for serie in series],
        linewidths=0,
        **kwargs
    )
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    ax.grid(False)
    # Exibindo o gráfico
    plt.show()

    return correlation_matrix


# this function evaluates a model on a given validation set for n time-steps
def eval_model(
    model,
    n,
    series,
    val_series,
    target_series=None,
    scaler=None,
    detrend=None,
    returned=[],
    historical=False,
    plot=True,
):
    if target_series != None:
        pred_series = model.predict(n=n, series=target_series)
    else:
        pred_series = model.predict(n=n)
    if historical:
        historical_series = model.historical_forecasts(target_series)
    else:
        historical_series = None
    if scaler != None:
        pred_series = pred_series.fit_inverse_transform(scaler)
        if historical:
            historical_series = historical_series.fit_inverse_transform(scaler)
        if target_series != None:
            target_series = target_series.fit_inverse_transform(scaler)
    if detrend != None:
        pred_series = pred_series.fit_inverse_detrend(detrend)
        if historical:
            historical_series = historical_series.fit_inverse_detrend(detrend)
        if target_series != None:
            target_series = target_series.fit_inverse_detrend(detrend)
    try:
        mape_val = mape(val_series, pred_series[: len(val_series)])
    except:
        mape_val = float("inf")
    try:
        sle_val = sle(val_series, pred_series[: len(val_series)])
    except:
        sle_val = float("inf")
    if historical:
        try:
            mape_train = mape(
                target_series[-len(historical_series) :], historical_series
            )
        except:
            mape_train = float("inf")
        try:
            sle_train = sle(target_series[-len(historical_series) :], historical_series)
        except:
            sle_train = float("inf")
    if plot:
        plt.figure(figsize=FIG_SIZE)
        series.plot(label="actual")
        pred_series.plot(label="forecast")
        if historical:
            historical_series.plot(label="historical")
            plt.title(
                "MAPE: t{:.2f}%".format(mape_train)
                + " v{:.2f}%".format(mape_val)
                + " - SLE: t{:.2f}".format(sle_train)
                + " v{:.2f}".format(sle_val)
            )
        else:
            plt.title(
                "MAPE: {:.2f}%".format(mape_val) + " - SLE: {:.2f}".format(sle_val)
            )
        plt.legend()
        plt.show()
    try:
        returned_f = []
        for returned_l in returned:
            if returned_l.upper() == "MAPE_VAL":
                returned_f.append(mape_val)
            elif returned_l.upper() == "SLE_VAL":
                returned_f.append(sle_val)
            elif returned_l.upper() == "MAPE_TRAIN":
                returned_f.append(mape_train)
            elif returned_l.upper() == "SLE_TRAIN":
                returned_f.append(sle_train)
            elif returned_l.upper() == "PREDICT_VALUES":
                returned_f.append(pred_series)
            elif returned_l.upper() == "HISTORICAL_VALUES":
                returned_f.append(historical_series)
            else:
                returned_f.append(None)
        return returned_f
    except:
        return None


def calculate_dates_diff(start, end, freq="D"):
    date_range = pd.date_range(start=start, end=end, freq=freq)
    return len(date_range)


def set_pl_trainer_kwargs(**kwargs):
    pl_trainer_kwargs = kwargs

    if torch.cuda.is_available():
        try:
            pl_trainer_kwargs["accelerator"]
        except:
            pl_trainer_kwargs["accelerator"] = "gpu"
        try:
            pl_trainer_kwargs["devices"]
        except:
            pl_trainer_kwargs["devices"] = -1
    else:
        pl_trainer_kwargs["accelerator"] = "cpu"

    if pl_trainer_kwargs["accelerator"] == "cpu":
        try:
            del pl_trainer_kwargs["devices"]
        except:
            pass

    return pl_trainer_kwargs


def is_decreasing(array):
    return array == sorted(array, reverse=True)


def info_message(message):
    cor_vermelha = "\033[91m"
    reset_cor = "\033[0m"

    # Imprime a mensagem em vermelho
    print(f"{cor_vermelha}INFO:   {reset_cor}{message}")


def get_first_week_day_for_month(day, year, month):
    for dia in range(1, 10):
        start_date = datetime(year, month, dia)
        dia_da_semana = start_date.strftime("%A")
        if dia_da_semana == day:
            return start_date


def count_month_week_days(day, year, month):
    count = 0
    for i in pd.date_range(
        start="{}-{}-{}".format(
            year, month, get_first_week_day_for_month(day, year, month).day
        ),
        end="{}-{}-01".format(
            year if month < 12 else year + 1, month + 1 if month < 12 else 1
        ),
        freq="W-MON",
    ):
        count += 1 if i.month == month else 0
    return count


def resample_month_series_in_week_series(day, columns, df, date_index):
    dates = pd.date_range(
        start=get_first_week_day_for_month(
            day, df[date_index][0].year, df[date_index][0].month
        ),
        end=get_first_week_day_for_month(
            day, df[date_index][len(df) - 1].year, df[date_index][len(df) - 1].month
        ),
        freq="W-MON",
    )
    date = 0
    data_columns = np.zeros((len(columns), len(dates)))
    for i in range(1, len(df)):
        n_dates = count_month_week_days(
            day, df[date_index][i - 1].year, df[date_index][i - 1].month
        )
        actual_data = [df[column][i - 1] for column in columns]
        interval = [(df[column][i] - df[column][i - 1]) / n_dates for column in columns]
        for j in range(n_dates):
            for k in range(len(actual_data)):
                data_columns[k][date] = actual_data[k]
                actual_data[k] += interval[k]
            date += 1
    for k in range(len(actual_data)):
        data_columns[k][date] = actual_data[k]
        actual_data[k] += interval[k]
    # Criar um DataFrame com a coluna de datas
    df = pd.DataFrame({"Data": pd.to_datetime(dates)})
    # Adicionar dinamicamente as colunas ao DataFrame
    for nome_coluna, dados_coluna in zip(columns, data_columns):
        df[nome_coluna] = dados_coluna
    return df
