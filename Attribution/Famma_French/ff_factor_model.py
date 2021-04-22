'''
calculate alpha and beta of portfolio against Famma French Factors
'''
import warnings
warnings.filterwarnings("ignore")

import sys,os

import pandas as pd
import pandas_datareader as web
import yfinance as yf

import statsmodels.api as smf
import urllib.request
import zipfile

from datetime import date

import matplotlib.pyplot as plt 

'''preprocess'''
cwd = os.getcwd()
iodir = f'{cwd}/io'
data_dir= f'{iodir}/data/'

parse_dir = {'F-F_Momentum_Factor_daily_CSV':(13, None,'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip'),
            'F-F_Research_Data_Factors_daily_CSV':(4, None,'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'),
            'F-F_ST_Reversal_Factor_daily_CSV':(4, None, 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_ST_Reversal_Factor_daily_CSV.zip'),
            'F-F_LT_Reversal_Factor_daily_CSV':(13, None, 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_LT_Reversal_Factor_daily_CSV.zip')}
            # name: (skiprows, splitrow_str, zip_path)

def get_fama_french(datafile):
    skiprows, splitrow_str, url = parse_dir.get(datafile)
    name = datafile #url.split('.', -1)[-2].split('/',-1)[-1] # extract file name from url
    print(f"Parsing {name}: SkipRows:{skiprows}, Split@{splitrow_str}")

    ''' retrieve & extract .zip data '''
    localp_zip = f'{iodir}/data/{name}.zip'
    urllib.request.urlretrieve(url,localp_zip)
    zip_file = zipfile.ZipFile(localp_zip, 'r')
    zip_file.extractall(data_dir)
    zip_file.close()
    csv_name = name[:-4] #drop the _CSV included in zip name

    ''' read local data '''
    localp_csv = f'{iodir}/data/{csv_name}.CSV'
    ff_factors = pd.read_csv(localp_csv, skiprows = skiprows, index_col = 0).reset_index().dropna().rename(columns={'index':'date'})
    ff_factors['date']= pd.to_datetime(ff_factors['date'], format= '%Y%m%d')

    if splitrow_str != None: # daily files do not need be split, monthly files also contain annual aggregations and must be split off the first annual index i.e. '  1927'
        split_index = ff_factors.loc[ff_factors['date'] == splitrow_str].index.values[0] 
        print(split_index)
        msplit = split_index - 2 
        ff_factors = ff_factors[:msplit]
        # ff_factors['date'] = ff_factors['date'] + pd.offsets.MonthEnd() # offset for monthly data  #pd.offsets.YearEnd()

    ff_factors = ff_factors.set_index('date').dropna()

    def fx(x):
        return x/100 # convert values from percentages to decimals
    for c in ff_factors.columns:
        ff_factors[c] = pd.to_numeric(ff_factors[c]).apply(fx)
    print(ff_factors.tail())

    return ff_factors


get_fama_french('F-F_LT_Reversal_Factor_daily_CSV')



# '''analysis'''

# today = date.today()
# # @ todo: import performance.returns to calculate time series of portfiolio returns instead of single stock adj close
# returns = yf.download('AAPL','1975-01-01',today)['Adj Close'].resample('M').last().pct_change()
# print(returns.head(2))

# # merge portfolio performance/daily returns data
# merge_df = monthly_ff.merge(returns, left_index = True, right_index = True, how = 'inner')

# merge_df.rename(columns={"Adj Close": "Portfolio Returns", "Mkt-RF":"mkt_excess"}, inplace=True)
# merge_df['port_excess'] = merge_df['Portfolio Returns'] - merge_df['RF']
# print(merge_df.tail(5))

# def visualize(merge_df):
#     ((merge_df +1).cumprod()).plot(color = ['b','r','g','y','pink','orange'], figsize=(15, 7))
    
#     plt.title(f"Returns for FF Factors", fontsize=16)
#     plt.ylabel('Cumulative Returns', fontsize=14)
#     plt.xlabel('Year', fontsize=14)
#     plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
#     plt.legend()
#     plt.yscale('log')


#     plt.show()

# visualize(merge_df)

# ''' multiple regression model '''
# model = smf.formula.ols(formula = "port_excess ~ mkt_excess + SMB + HML", data = merge_df).fit()

# print(model.summary())
# print('Parameters: ', model.params)
# print('R2: ', model.rsquared)