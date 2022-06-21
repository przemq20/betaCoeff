import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import register_matplotlib_converters
from beta import read_csv

register_matplotlib_converters()

if __name__ == '__main__':
    path = 'data/wig/beta/10/MIL.csv'

    series = read_csv(path, lambda row: row[0])
    series = list(map(lambda x: float(x), filter(lambda x: x != 'nan', series)))

    split_idx = (len(series) * 3) // 4
    df = pd.DataFrame(data={'beta': series})

    train = df.beta[:split_idx]
    test = df.beta[split_idx:]


    # best_aic = 1e16

    model = SARIMAX(train, order=(3,0,3), seasonal_order=(2,1,0,12))
    best_results = model.fit()
    predictions = best_results.predict()  #
    forecast = best_results.forecast(len(test), alpha=0.05)

    plt.plot(df)
    plt.plot(predictions)
    plt.plot(forecast, c='r')
    plt.savefig('seasonal.png')
    plt.show()
