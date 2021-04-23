import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 



def interpolate_na(data=None, col_name=None, method='time'):
    data[col_name] = data[col_name].interpolate(method, axis = 0) #linear or time
    return data

def smoothing(data=None, col_name=None,):
    # data[col_name] = data.groupby([data.index.date]).apply(lambda x: pd.ewm(x, alpha=0.84))
    #kalman

    # seasonal
    print(data.head())
    print(type(data.index))
    from statsmodels.tsa.seasonal import seasonal_decompose
    print(data[col_name])
    # data = data.sort_index(inplace=True)
    result = seasonal_decompose(data[col_name], model='additive')
    print(result.trend)
    print(result.seasonal)
    print(result.resid)
    print(result.observed)

    return data

def check_stationarity(data=None, col_name=None,):
    #https://machinelearningmastery.com/time-series-data-stationary-python/
    X = data[col_name].values
    split = round(len(X) / 2)
    X1, X2 = X[0:split], X[split:]
    mean1, mean2 = X1.mean(), X2.mean()
    var1, var2 = X1.var(), X2.var()
    print('mean1=%f, mean2=%f' % (mean1, mean2))
    print('variance1=%f, variance2=%f' % (var1, var2))
    data[col_name].hist()
    plt.show()

    X = np.log(X) #log transform
    plt.hist(X)
    plt.show()


def self_correlation(data=None, col_name=None,):
    from statsmodels.graphics import tsaplots

    # Display the autocorrelation plot of your time series
    fig = tsaplots.plot_acf(data[col_name], lags=24)
    plt.show()

    # spurious & lagging correlation??
    # cointegration


def autoregression(data=None, col_name=None,):
    #https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
    from pandas import read_csv
    from matplotlib import pyplot
    from statsmodels.tsa.ar_model import AutoReg
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    # load dataset
    series = data[col_name]
    # split dataset
    X = series.values
    train, test = X[1:len(X)-7], X[len(X)-7:]
    # train autoregression
    model = AutoReg(train, lags=29)
    model_fit = model.fit()
    print('Coefficients: %s' % model_fit.params)
    # make predictions
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    for i in range(len(predictions)):
        print('predicted=%f, expected=%f' % (predictions[i], test[i]))
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
    # plot results
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()


def arima(data=None, col_name=None,):
    #https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
    from pandas import read_csv
    from pandas import datetime
    from matplotlib import pyplot
    from pandas.plotting import autocorrelation_plot
    from statsmodels.tsa.arima.model import ARIMA

    def parser(x):
        return datetime.strptime('190'+x, '%Y-%m')
    
    series = data[col_name]
    autocorrelation_plot(series)
    plt.show()
    series.index = series.index.to_period('M')
    # fit model
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    # summary of fit model
    print(model_fit.summary())
    # line plot of residuals
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    # density plot of residuals
    residuals.plot(kind='kde')
    plt.show()
    # summary stats of residuals
    print(residuals.describe())

    #seasonal ARIMA; ARCH; GARCH
    