
import warnings
warnings.filterwarnings('ignore')
import sys, os

import seaborn as sns
import pandas_datareader as web
import statsmodels.tsa.api as tsa
import pandas as pd
import numpy as np 
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, acf, plot_pacf, pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
import statsmodels.api as sm
from scipy.stats import probplot, moment
from sklearn.metrics import mean_squared_error

p = os.getcwd() +'/io/'
'''
@ original source from:
-pakt publishing ML for algorithmic trading; Stefan Jansen-
https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition/blob/master/09_time_series_models/02_arima_models.ipynb
Univariate time series models
"Univariate time series models relate the value of the time series at the point in time of interest to a linear combination of lagged values "
Autoregressive models
--

ARIMA = autoregressive integrated moving average: como of autoregression and moving averages to find complementary aspects 
seasonal autoregressive moving average model with exogenous inputs....
SARIMAX - Seasonal

We iterate over various (p, q) lag combinations and collect diagnostic statistics to compare the result.


'''
def plot_correlogram(x, lags=None, title=None):    
    lags = min(10, int(len(x)/5)) if lags is None else lags
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
    x.plot(ax=axes[0][0], title='Residuals')
    x.rolling(21).mean().plot(ax=axes[0][0], c='k', lw=1)
    q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
    stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f}'
    axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
    probplot(x, plot=axes[0][1])
    mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
    s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
    axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
    plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
    plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
    axes[1][0].set_xlabel('Lag')
    axes[1][1].set_xlabel('Lag')
    fig.suptitle(title, fontsize=14)
    sns.despine()
    fig.tight_layout()
    fig.subplots_adjust(top=.9)
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(f'{str(p)}{title}.png')



def univariate_time_series_model():
    industrial_production = web.DataReader('IPGMFN', 'fred', '1988', '2017-12').squeeze().dropna()
    industrial_production_log = np.log(industrial_production)
    industrial_production_log_diff = industrial_production_log.diff(12).dropna()


    model1 = tsa.statespace.SARIMAX(industrial_production_log, order=(2,0,2), seasonal_order=(0,1,0,12)).fit()
    model2 = tsa.statespace.SARIMAX(industrial_production_log_diff, order=(2,0,2), seasonal_order=(0,0,0,12)).fit()
    print(model1.params.to_frame('SARIMAX').join(model2.params.to_frame('diff')))

    ''' find optimal ARMA lags "We iterate over various (p, q) lag combinations and collect diagnostic statistics to compare the result." '''
    train_size = 120
    test_results = {}
    y_true = industrial_production_log_diff.iloc[train_size:]
    for p in range(5):
        for q in range(5):
            aic, bic = [], []
            if p == 0 and q == 0:
                continue
            print(p, q)
            convergence_error = stationarity_error = 0
            y_pred = []
            for T in range(train_size, len(industrial_production_log_diff)):
                train_set = industrial_production_log_diff.iloc[T-train_size:T]
                try:
                    model = tsa.ARMA(endog=train_set, order=(p, q)).fit()
                except LinAlgError:
                    convergence_error += 1
                except ValueError:
                    stationarity_error += 1

                forecast, _, _ = model.forecast(steps=1)
                y_pred.append(forecast[0])
                aic.append(model.aic)
                bic.append(model.bic)

            result = (pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
                    .replace(np.inf, np.nan)
                    .dropna())

            rmse = np.sqrt(mean_squared_error(
                y_true=result.y_true, y_pred=result.y_pred))

            test_results[(p, q)] = [rmse,
                                    np.mean(aic),
                                    np.mean(bic),
                                    convergence_error,
                                    stationarity_error]

    test_results = pd.DataFrame(test_results).T
    test_results.columns = ['RMSE', 'AIC', 'BIC', 'convergence', 'stationarity']
    test_results.index.names = ['p', 'q']
    test_results.info()
    test_results.dropna()
    print(test_results.nsmallest(5, columns=['RMSE']))
    print(test_results.nsmallest(5, columns=['BIC']))

    sns.heatmap(test_results.RMSE.unstack().mul(10), fmt='.2', annot=True, cmap='Blues_r')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(f'{str(p)}RMSE_heatmap.png')



    sns.heatmap(test_results.BIC.unstack(), fmt='.2f', annot=True, cmap='Blues_r')
    fig2 = plt.gcf()
    plt.show()
    fig2.savefig(f'{str(p)}BIC_heatmap.png')

    model = tsa.ARMA(endog=industrial_production_log_diff, order=(0, 4)).fit()
    print(model.summary())
    plot_correlogram(model.resid, title='arma_corr')


    best_p, best_q = test_results.rank().loc[:, ['RMSE', 'BIC']].mean(1).idxmin()
    best_arma_model = tsa.ARMA(endog=industrial_production_log_diff, order=(best_p, best_q)).fit()
    print(best_arma_model.summary())
    plot_correlogram(best_arma_model.resid, lags=20, title='Residuals_ARMA')
 


    
    best_model = tsa.SARIMAX(endog=industrial_production_log_diff, order=(2, 0, 3),
                            seasonal_order=(1, 0, 0, 12)).fit()
    print(best_model.summary())
    plot_correlogram(best_model.resid, lags=20, title='Residuals_SARIMAX')
 


univariate_time_series_model()