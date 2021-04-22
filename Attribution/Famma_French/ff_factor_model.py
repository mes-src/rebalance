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
from functools import reduce

from datetime import date

'''etc'''
cwd = os.getcwd()
iodir = f'{cwd}/io'
data_dir= f'{iodir}/data/'

parse_dict = {'F-F_Momentum_Factor_daily_CSV':(13, None,'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip'),
            'F-F_Research_Data_Factors_daily_CSV':(4, None,'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'),
            'F-F_ST_Reversal_Factor_daily_CSV':(13, None, 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_ST_Reversal_Factor_daily_CSV.zip'),
            'F-F_LT_Reversal_Factor_daily_CSV':(13, None, 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_LT_Reversal_Factor_daily_CSV.zip')}
            # name: (skiprows, splitrow_str, zip_path)


'''data'''

def get_fama_french(datafile):
    skiprows, splitrow_str, url = parse_dict.get(datafile)
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
        msplit = split_index - 2 
        ff_factors = ff_factors[:msplit]
        # ff_factors['date'] = ff_factors['date'] + pd.offsets.MonthEnd() # offset for monthly data  #pd.offsets.YearEnd()

    ff_factors = ff_factors.set_index('date').dropna()

    def fx(x):
        return x/100 # convert values from percentages to decimals
    for c in ff_factors.columns:
        ff_factors[c] = pd.to_numeric(ff_factors[c]).apply(fx)

    return ff_factors



def build_ff_factor_frame():
    frames = []
    for k,v in parse_dict.items():
        _ = get_fama_french(k)
        frames.append(_)
    df = reduce(lambda x, y: pd.merge(x, y, on = 'date'), frames)
    return df 
ffdf = build_ff_factor_frame()



def build_portfolio_returns():
    # import Performance Module to retrieve portfolio daily price returns
    import inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    sys.path.append(parentdir)

    from Performance import Returns as perfRet 
    returns = perfRet.portfolio_returns(arr = ['MSFT','AAPL'], weights = [0.5,0.5], date_start = '1975-01-01', date_end = date.today(), time_sample='D', cumulative= False, plot = False).rename({'Date':'date'})
    return returns
portfolio_returns = build_portfolio_returns()


df = ffdf.merge(portfolio_returns, left_index = True, right_index = True, how = 'inner')
df.rename(columns={"weighted_returns": "Portfolio Returns", "Mkt-RF":"mkt_excess", "Mom   ":"Mom"}, inplace=True)
df['port_excess'] = df['Portfolio Returns'] - df['RF']
print(df)





'''analysis'''

def factor_analysis(df, plot = False):
    def plot(df):
        ((merge_df +1).cumprod()).plot(color = ['b','r','g','y','pink','orange'], figsize=(15, 7))
        plt.title(f"Returns for FF Factors", fontsize=16)
        plt.ylabel('Cumulative Returns', fontsize=14)
        plt.xlabel('Year', fontsize=14)
        plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
        plt.legend()
        plt.yscale('log')
        plt.show()

    if plot == True:
        plot(df)

    ''' multiple regression model '''
    print(df.columns)
    model = smf.formula.ols(formula = "port_excess ~ mkt_excess + SMB + HML + ST_Rev + LT_Rev + Mom", data = df).fit()
    print(model.summary())
    print('Parameters: ', model.params)
    print('R2: ', model.rsquared)

factor_analysis(df)