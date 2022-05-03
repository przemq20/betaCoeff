import csv
import datetime
from collections import OrderedDict

import numpy

stocks = OrderedDict()
markets = OrderedDict()
combined = OrderedDict()


def dateReader(date_text):
    dt = datetime.datetime.strptime(date_text, '%Y-%m-%d').strftime('%y/%m/%d')
    return '20' + dt


def readStocks():
    with open('cmr_d2.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            stocks[dateReader(row[0])] = row[1]


def readMarkets():
    with open('wig_d.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            markets[dateReader(row[0])] = row[1]


def alignData():
    for stockdate in stocks:
        if stockdate in markets and not "N/A" in markets[stockdate]:
            combined[stockdate] = (stocks[stockdate], markets[stockdate])


# get 2 lists, stock prices and market value for given year
def getYearData(year):
    stocklist = []
    marketlist = []
    for date in combined:
        if year in date:
            stocklist.append(float(combined[date][0]))
            marketlist.append(float(combined[date][1]))
    return (stocklist, marketlist)


def calculateBeta(stocklist, marketlist):
    var = numpy.var(stocklist)
    covar = numpy.cov(stocklist, marketlist)[0][1] / 100.0  # norm
    print(covar, var)
    return covar / var


def main():
    readStocks()
    readMarkets()
    alignData()
    (sl, ml) = getYearData('2021')  # Enter the year here
    beta = calculateBeta(sl, ml)
    print(beta)


main()
