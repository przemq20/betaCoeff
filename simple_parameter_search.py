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

    best_aic = 1e16
    best_p = 0
    best_q = 0
    best_results = None
    for p in range(1, 11):
        for q in range(1, 11):
            model = ARIMA(train, order=(p, 0, q))  # create model
            results = model.fit()  # fit model t parameters
            if results.aic < best_aic:
                best_aic = results.aic
                best_results = results
                best_q = q
                best_p = p

    print(best_q, ' ', best_p, ' ', best_aic)
    predictions = best_results.predict()  #
    forecast = best_results.forecast(len(test), alpha=0.05)

    plt.plot(df)
    plt.plot(predictions)
    plt.plot(forecast, c='r')
    plt.savefig('parameter_search.png')
    plt.show()
