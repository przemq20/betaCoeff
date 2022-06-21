import os

import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
from beta import read_csv
from sklearn.metrics import mean_squared_error
import numpy as np

register_matplotlib_converters()

if __name__ == '__main__':
    path = 'data/wig/beta/10/'

    errors = []
    aics = []
    for file in os.listdir(path):
        series = read_csv(path + file, lambda row: row[0])
        series = list(map(lambda x: float(x), filter(lambda x: x != 'nan', series)))

        split_idx = (len(series) * 3) // 4
        df = pd.DataFrame(data={'beta': series})

        train = df.beta[:split_idx]
        test = df.beta[split_idx:]
        p = 3
        q = 3

        model = ARIMA(train, order=(p,0,q))
        results = model.fit()
        predictions = results.predict()  #
        forecast = results.forecast(len(test), alpha=0.05)

        errors.append(np.sqrt(mean_squared_error(test, forecast)))
        aics.append(results.aic)

    print('Mean error: ', sum(errors) / len(errors))
    print('Mean AIC: ', sum(aics) / len(aics))

