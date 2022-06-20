import pandas as pd
from matplotlib import pyplot as plt
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import register_matplotlib_converters
from beta import read_csv
import pmdarima as pm
from pmdarima import model_selection
from sklearn.metrics import mean_squared_error
import numpy as np
from pmdarima.arima.stationarity import ADFTest

register_matplotlib_converters()

if __name__ == '__main__':
    path = 'data/wig/beta/10/MIL.csv'

    series = read_csv(path, lambda row: row[0])
    series = list(map(lambda x: float(x), filter(lambda x: x != 'nan', series)))

    split_idx = (len(series) * 3) // 4
    df = pd.DataFrame(data={'beta': series})

    train = df.beta[:split_idx]
    test = df.beta[split_idx:]

    # Fit a simple auto_arima model
    modl = pm.auto_arima(train, start_p=1, start_q=1, start_P=0, start_Q=0,
                         max_p=5, max_q=5, max_P=0, max_Q=0, seasonal=False,
                         stepwise=True, suppress_warnings=True, D=0, max_D=0,
                         error_action='ignore')

    # Create predictions for the future, evaluate on test

    preds, conf_int = modl.predict(n_periods=test.shape[0], return_conf_int=True)

    # print(preds)
    # Print the error:
    print("Test RMSE: %.3f" % np.sqrt(mean_squared_error(test, preds)))

    # #############################################################################
    # Plot the points and the forecasts
    x_axis = np.arange(train.shape[0] + preds.shape[0])

    plt.plot(x_axis[:train.shape[0]], train, alpha=0.75)
    plt.plot(x_axis[train.shape[0]:], preds, alpha=0.75)  # Forecasts
    plt.plot(x_axis[train.shape[0]:], test,
                alpha=0.4)  # Test data
    plt.fill_between(x_axis[-preds.shape[0]:],
                     conf_int[:, 0], conf_int[:, 1],
                     alpha=0.1, color='b')
    plt.show()
