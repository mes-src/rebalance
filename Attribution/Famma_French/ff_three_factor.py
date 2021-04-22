'''
calculate alpha and beta of portfolio against Famma French 3 Factor Model
'''

import pandas as pd
import pandas_datareader as web
import yfinance as yf

import statsmodels.api as smf
import urllib.request
import zipfile

from datetime import date

import matplotlib.pyplot as plt 

'''preprocess'''

def get_fama_french():
    # Web url
    ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
    # Download the file and save it
    # We will name it fama_french.zip file
    urllib.request.urlretrieve(ff_url,'fama_french.zip')
    zip_file = zipfile.ZipFile('fama_french.zip', 'r')
    # Next we extact the file data
    zip_file.extractall()
    # Make sure you close the file after extraction
    
    zip_file.close()
    

    ff_factors = pd.read_csv('F-F_Research_Data_Factors.csv', skiprows = 3, index_col = 0).reset_index().dropna().rename(columns={'index':'date'}) 
    split_index = ff_factors.loc[ff_factors['date'] == '  1927'].index.values[0] # 1927 is the first year that annual figures are provided for; split dataframes at this location

    # @ to do: clean both datasets with simple function

    msplit = split_index - 2
    monthly_df = ff_factors[:msplit]
    monthly_df['date']= pd.to_datetime(monthly_df['date'], format= '%Y%m')
    monthly_df['date'] = monthly_df['date'] + pd.offsets.MonthEnd()
    monthly_df = monthly_df.set_index('date').dropna()
    def fx(x):
        return x/100 # convert values from percentages to decimals
    for c in monthly_df.columns:
        monthly_df[c] = pd.to_numeric(monthly_df[c]).apply(fx)


    asplit = split_index
    annual_df = ff_factors[asplit:]
    annual_df['date']= pd.to_datetime(annual_df['date'].map(lambda x : x.strip()), format= '%Y') # format annual dates to year end
    annual_df['date'] = annual_df['date'] + pd.offsets.YearEnd()
    annual_df = annual_df.set_index('date').dropna()
    for c in annual_df.columns:
        annual_df[c] = pd.to_numeric(annual_df[c]).apply(fx)

    return monthly_df, annual_df

monthly_ff, annual_ff = get_fama_french()




'''analysis'''

today = date.today()

returns = yf.download('AAPL','1975-01-01',today)['Adj Close'].resample('M').last().pct_change()
print(returns.head(2))

# merge portfolio performance/daily returns data
merge_df = monthly_ff.merge(returns, left_index = True, right_index = True, how = 'inner')

merge_df.rename(columns={"Adj Close": "Portfolio Returns", "Mkt-RF":"mkt_excess"}, inplace=True)
merge_df['port_excess'] = merge_df['Portfolio Returns'] - merge_df['RF']
print(merge_df.tail(5))

def visualize(merge_df):
    ((merge_df +1).cumprod()).plot(color = ['b','r','g','y','pink','orange'], figsize=(15, 7))
    
    plt.title(f"Returns for FF Factors", fontsize=16)
    plt.ylabel('Cumulative Returns', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    plt.legend()
    plt.yscale('log')


    plt.show()

visualize(merge_df)

''' multiple regression model '''
model = smf.formula.ols(formula = "port_excess ~ mkt_excess + SMB + HML", data = merge_df).fit()

print(model.summary())
print('Parameters: ', model.params)
print('R2: ', model.rsquared)