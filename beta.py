import csv
import numpy

periods_in_days = [ 5, 10, 20, 40, 60]


def calculate_beta(stock_markings, market_markings):
    var = numpy.var(stock_markings)
    covar = numpy.cov(stock_markings, market_markings)[0][1] / 100.0  # norm

    return covar / var


def read_csv(file_name, map_row):
    markings = []
    with open(file_name, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)  # drop line with names
        for row in reader:
            markings.append(map_row(row))
    return markings


def split_in_chunks(list_to_split, n):
    return [list_to_split[i:i+n] for i in range(0, len(list_to_split), n)]


def transform_date(date):
    date = str(date)
    return f'{date[:4]}-{date[4:6]}-{date[6:]}'


if __name__ == "__main__":
    index_markings_with_dates = read_csv('data/wig/index_markings/wig_d.csv', lambda row: (row[0], float(row[4])))
    tickers = read_csv('data/wig/components/components.csv', lambda row: row[1])

    for ticker in tickers:
        print('\n')
        try:
            stock_markings_with_dates = read_csv(f'data/wig/components_markings/{ticker.lower()}.txt',
                                                 lambda row: (transform_date(row[2]), float(row[7])))
            dates = set(map(lambda x: x[0], stock_markings_with_dates))
            trimmed_index_markings_with_dates = list(filter(lambda x: x[0] in dates, index_markings_with_dates))
            stocks_markings = list(map(lambda x: x[1], stock_markings_with_dates))
            index_markings = list(map(lambda x: x[1], trimmed_index_markings_with_dates))
            try:
                for period in periods_in_days:
                    print(f'{ticker} - {period}')
                    stocks_markings_parts = split_in_chunks(stocks_markings, period)
                    index_markings_parts = split_in_chunks(index_markings, period)

                    with open(f'data/wig/beta/{period}/{ticker}.csv', 'w') as file:
                        file.write('beta\n')
                        for s, i in zip(stocks_markings_parts, index_markings_parts):
                            beta = calculate_beta(s, i)
                            file.write(f'{beta}\n')
            except Exception:
                pass
        except Exception:
            pass
