import pandas as pd
from statsmodels.tsa.stattools import adfuller

from beta import read_csv
import os

if __name__ == '__main__':
    path = 'data/wig/beta'
    periods = os.listdir(path)

    for period in periods:
        files = os.listdir(f'{path}/{period}')
        total = 0
        passed = 0
        too_short = set()
        non_stationary = set()

        for file in files:
            series = read_csv(f'{path}/{period}/{file}', lambda row: row[0])
            series = list(map(lambda x: float(x), filter(lambda x: x != 'nan', series)))
            series_len = len(series)

            if series_len < 50:
                too_short.add(file)
            else:
                total += 1
                df = pd.DataFrame(data={'beta': series})
                result = adfuller(df['beta'])

                if result[1] <= 0.05 and len(list(filter(lambda x: x > result[0], result[4].values()))) > 0:
                    passed += 1
                else:
                    non_stationary.add(file)

        print(f'{period} - {passed} / {total}')
        for file_too_short in too_short:
            os.remove(f'{path}/{period}/{file_too_short}')

        for ns in non_stationary:
            os.remove(f'{path}/{period}/{ns}')

