# %reset -f
#----------------------------- Importing packages ----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os
from scipy.stats import norm
import matplotlib.dates as mdates

# Importing OSEBX Benchmark
OSEBX = pd.read_excel('01 Data/01 Input data/OSEBX_index 1996-2018.xlsx').iloc[190:4962, 1:2].values
OSEBX = np.flipud(OSEBX)
real_return = pd.read_csv('01 Data/02 Preprocessed data/daily real return OSEBX_proc.csv', header = None)
binary_returns = pd.read_csv('01 Data/02 Preprocessed data/daily binary return OSEBX_proc.csv', header = None) # 1 if above cross-sectional median return, else 0
real_return, binary_returns = np.array(real_return), np.array(binary_returns)

# Adding correct dates as X-axis for graphs
dates = pd.read_csv('01 Data/02 Preprocessed data/daily prices OSEBX_proc.csv', sep = ',', header = None).iloc[772:5523, 0] 
dates = pd.to_datetime(dates, format="%Y-%m-%d")
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')


# Making list of all prediction outputs
files_prob = sorted(glob.glob(os.path.join('04 Predictions/01 Chapter 5 Analysis and Results/01 prob', "*.csv")))       
files_class = sorted(glob.glob(os.path.join('04 Predictions/01 Chapter 5 Analysis and Results/02 class', "*.csv")))

pred_prob = pd.read_csv(files_prob[4], header = None) # Extracting optimal model
pred_prob = np.array(pred_prob)

# Function calculates cumulative return after transaction costs
#
## Takes predictions (prob), number of stocks (k) and transaction costs(tx)
def aftertx(prob, k, tx):   
    #tx = 0.00029
    prob = np.array(prob)
    long10 = [100]
    l10idxx = []
    weight = []
    for i in range(0, len(pred_prob)):
        a = pred_prob[i,]
        b = a[~np.isnan(a)]
        e = b[np.argsort(b)[-k:]]   # top k probabilities
        l10idx = np.where(np.isin(a,e))[0].tolist()
        l10idxx.append(l10idx)  
        
        del_list = []
        for s in range(0, len(l10idxx[i])):
            if str(real_return[i, l10idxx[i][s]]) == 'nan':
                del_list.append(l10idxx[i][s])
        
        if len(del_list) != 0:
            for x in range(0, len(del_list)):
                l10idxx[i][:] = (value for value in l10idxx[i] if value != del_list[x])
        
        if len(l10idxx[i][:]) == 0:
            weight.append(0)
        if len(l10idxx[i][:]) != 0:
            weights_t=[1/len(l10idxx[i])]*len(l10idxx[i])
            weight.append(weights_t)
        
    for t in range(0, len(pred_prob)):
        buy = []    # at t=0, wether to buy
        sell = []   # at t = 1, whether to sell
        
        if t == 0:
            for s in l10idxx[t]:
                if s not in l10idxx[t+1]:
                    sell.append(s)
                buy.append(s)
    
        if (t>0) and (t<4749):
            for s in l10idxx[t]:
                if s not in l10idxx[t+1]:
                    sell.append(s)
            for k in l10idxx[t]:
                if k not in l10idxx[t-1]:
                    buy.append(k)
        
        if t == 4749:
            for k in l10idxx[t]:
                if k not in l10idxx[t-1]:
                    buy.append(k)
                sell.append(k)
        
        power = np.array(l10idxx[t])
        power[:] = 0
        for x in range(0,len(power)):
            if l10idxx[t][x] in buy:
                power[x] += 1
            if l10idxx[t][x] in sell:
                power[x] += 1
        
        if len(l10idxx[t]) == 0:
            temp = long10[t]
        if len(l10idxx[t]) != 0:
            temp = sum((long10[t]*np.array(weight[t]))*(1+real_return[t,l10idxx[t]])*((1-(tx))**power))
            
        long10.append(temp)
    return long10

# Function calculates max drawdown
def max_drawdown(X):
    mdd = 0
    peak = X[0]
    for x in X:
        if x > peak: 
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
    return mdd

# Function calculates return performance
def returnstats(portfolio):
    ret = ((portfolio / np.roll(portfolio, 1, axis = 0)) - 1 )[1:]
    tot_ret = portfolio[len(portfolio)-1]/portfolio[0] - 1
    annual_ret = (1+tot_ret)**(1/(len(portfolio)/250)) - 1
    annual_std = np.std(ret) * np.sqrt(250)
    annual_sharpe = (annual_ret-0.0325)/annual_std
    mdd = max_drawdown(portfolio)
    var95 = norm.ppf(1-0.95, annual_ret, annual_std)
    list = [annual_ret, annual_std, annual_sharpe, mdd, var95]
    return list

# Results:

# OSEBX index
osebx_portfolio = [100]
OSEBX_return = ((OSEBX / np.roll(OSEBX, 1, axis = 0)) - 1 )[1:]
for i in range(0, len(OSEBX_return)):
    osebx_portfolio.append(osebx_portfolio[i]*(1+OSEBX_return[i,0]))
osebx_portfolio = osebx_portfolio[:4751]

# Our optimal model: LSTM_a with K = 5
long5 = aftertx(pred_prob, 5, tx = 0.00029)

## Making table of results
table = []
table.append(returnstats(long5))
table.append(returnstats(osebx_portfolio)) 

table.append(returnstats(long5[:2512]))          # before financial crisis
table.append(returnstats(osebx_portfolio[:2512]))

table.append(returnstats(long5[2512:]))          # after financial crisis
table.append(returnstats(osebx_portfolio[2512:]))

table.append(returnstats(long5[4018:]))          # last 3y
table.append(returnstats(osebx_portfolio[4018:]))

table = pd.DataFrame(table)
table.columns = ['Return', 'Standard deviation', 'Sharpe Ratio', 'Max drawdown', 'VaR 5\%']
table.index = ['LSTM_opt full period', 'OSEBX full period', 'lstm 99-08', 'osebx 99-08', 'lstm 09-17', 'osebx 09-17', 'lstm 15-17', 'osebx 15-17'] 
table = round(table, 4)
table = np.transpose(table)
#table.to_csv('Return table afte tx best vs osebx.csv', index=True, header=True, sep=',')

# Plot cumulative returns before transaction costs logs
fig, ax = plt.subplots()
line5, = ax.plot(dates, long5, color = 'royalblue', label='LSTM_a')
line7, = ax.plot(dates, osebx_portfolio, color = 'black', label='OSEBX')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yscale('log')
ax.legend(loc=2)
ax.set_ylabel(r"Portfolio value", labelpad=5)
ax.set_xlabel(r"Time", labelpad=5)
textstr = '\n'.join((
    r'$r=89\%$',
    r'$\sigma=31\%$',
    r'$Sharpe=2.72$'))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.75, 0.80, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

#plt.savefig('model mot osebx etter tx.png', bbox_inches='tight', dpi=500)

# Before 2009 (financial crisis). First trade day 2009 = 2512
fig, ax = plt.subplots()
line5, = ax.plot(dates[:2512], long5[:2512], color = 'royalblue', label='LSTM_a')
line7, = ax.plot(dates[:2512], osebx_portfolio[:2512], color = 'black', label='OSEBX')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yscale('log')
ax.set_ylim([0,max(long5[:2512])])
ax.legend(loc=2)
# textbox
textstr = '\n'.join((
    r'$r=95\%$',
    r'$\sigma=33\%$',
    r'$Sharpe=2.79$'))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.75, 0.80, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

ax.set_ylabel(r"Portfolio value", labelpad=5)
ax.set_xlabel(r"Time", labelpad=5)
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
#plt.savefig('model mot osebx etter tx 2009.png', bbox_inches='tight', dpi=500)

# After 2009 (financial crisis). First trade day 2009 = 2512
fig, ax = plt.subplots()
line5, = ax.plot(dates[2512:], (long5[2512:]/(0.01*long5[2512])), color = 'royalblue', label='LSTM_a')
line7, = ax.plot(dates[2512:], (osebx_portfolio[2512:]/(0.01*osebx_portfolio[2512])), color = 'black', label='OSEBX')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yscale('log')
ax.set_ylim([0,max(long5[:2512])])
ax.legend(loc=2)
# textbox
textstr = '\n'.join((
    r'$r=81\%$',
    r'$\sigma=30\%$',
    r'$Sharpe=2.60$'))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.75, 0.60, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

ax.set_ylabel(r"Portfolio value", labelpad=5)
ax.set_xlabel(r"Time", labelpad=5)
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
#plt.savefig('model mot osebx etter tx 2017.png', bbox_inches='tight', dpi=500)

# Creating table of performance for the 4 differnt periods
performance = []
before_2009 = long5[:2512]
after_2009 = (long5[2512:]/(0.01*long5[2512]))
last_3y = (long5[4018:]/(0.01*long5[4018]))
performance.append(returnstats(long5))
performance.append(returnstats(before_2009))
performance.append(returnstats(after_2009))
performance.append(returnstats(last_3y))

performance = pd.DataFrame(performance)
performance.index = ['1999-2018', '1999-2009', '2009-2018', '2015-2018']
performance.columns = ['Return', 'Standard deviation', 'Sharpe ratio', 'Max drawdown', 'VaR 5%']
performance = np.transpose(performance)
performance = round(performance, 4)

#performance.to_csv('performance 4 periods tx.csv', sep=',')

'''# Last three years (2015-). First trade day 2015 = 4018
fig, ax = plt.subplots()
line5, = ax.plot(dates[4018:], (long5[4018:]/(0.01*long5[4018])), color = 'royalblue', label='LSTM_a')
line7, = ax.plot(dates[4018:], (osebx_portfolio[4018:]/(0.01*osebx_portfolio[4018])), color = 'black', label='OSEBX')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#plt.yscale('log')
ax.legend(loc=2)
ax.set_ylabel(r"Portfolio value", labelpad=5)
ax.set_xlabel(r"Time", labelpad=5)
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)
plt.savefig('model mot osebx etter tx 3y.png', bbox_inches='tight', dpi=500)'''
