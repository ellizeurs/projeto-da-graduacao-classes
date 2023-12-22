import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import torch

from TimeSeries import TimeSeries
from darts.metrics import mape

from const import FIG_SIZE
from metrics.sle import sle

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