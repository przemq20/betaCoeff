import pandas as pd
from matplotlib import pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
from beta import read_csv

register_matplotlib_converters()

if __name__ == '__main__':
    path = 'data/wig/beta/10/MIL.csv'

    series = read_csv(path, lambda row: row[0])
    series = list(map(lambda x: float(x), filter(lambda x: x != 'nan', series)))

    df = pd.DataFrame(data={'beta': series})

    model = ARIMA(df, order=(1, 0, 2))  # create model
    results = model.fit()  # fit model to parameters

    predictions = results.predict()  #
    forecast = results.forecast(50, alpha=0.05)

    plt.plot(predictions)
    plt.plot(forecast)
    plt.show()
