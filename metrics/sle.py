import math
import numpy as np

# squared_log_error
def sle(y_true, y_pred):
    y_true_num = []
    y_pred_num = []

    for i in y_true.values():
        if math.isnan(i) == False:
            y_true_num.append(float(i))

    for i in y_pred.values():
        if math.isnan(i) == False:
            y_pred_num.append(float(i))

    somatorio = 0
    for i in range(0, len(y_true_num)):
        x = y_pred_num[i] / y_true_num[i]
        somatorio += (np.log(abs(x))) ** 2
    return somatorio