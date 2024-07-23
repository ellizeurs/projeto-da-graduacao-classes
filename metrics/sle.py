from darts.metrics import sle as sle_darts

def sle(series, pred_series):
    return np.mean(sle_darts(series, pred_series))