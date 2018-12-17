# %reset -f
#----------------------------- Importing packages ----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os
from scipy.stats import norm

# Importing data
OSEBX = pd.read_excel('01 Data/01 Input data/OSEBX_index 1996-2018.xlsx').iloc[190:4962, 1:2].values
OSEBX = np.flipud(OSEBX)
real_return = pd.read_csv('01 Data/02 Preprocessed data/daily real return OSEBX_proc.csv', header = None)
binary_returns = pd.read_csv('01 Data/02 Preprocessed data/daily binary return OSEBX_proc.csv', header = None) # 1 if above cross-sectional median return, else 0
real_return, binary_returns = np.array(real_return), np.array(binary_returns)

# Adding correct dates as X-axis for graphs
dates = pd.read_csv('01 Data/02 Preprocessed data/daily prices OSEBX_proc.csv', sep = ',', header = None).iloc[772:5523, 0] 
dates = pd.to_datetime(dates, format="%Y-%m-%d")

# Making list of all prediction outputs
files_prob = sorted(glob.glob(os.path.join('04 Predictions/01 Chapter 5 Analysis and Results/01 prob', "*.csv")))       
files_class = sorted(glob.glob(os.path.join('04 Predictions/01 Chapter 5 Analysis and Results/02 class', "*.csv")))

# Making empty table to append each result in
table = []
graph= []

# Start loop over models
for x in range(0, len(files_prob)):
    
    # Importing predictions, real_returns, binary_matrix
    pred_prob = pd.read_csv(files_prob[x], header = None)
    pred_class = pd.read_csv(files_class[x], header = None)
    
    pred_prob, pred_class, real_return = np.array(pred_prob), np.array(pred_class), np.array(real_return)
    
    #---------------------------- Trading strategy ------------------------------
    # Each strategy starts with 100 NOK
    long5 = [100]
    long10 = [100]
    long20 = [100]
    
    for i in range(0, len(pred_prob)):
        a = pred_prob[i,]
        b = a[~np.isnan(a)]
        c = b[np.argsort(b)[-5:]]   # top 5 probabilities
        e = b[np.argsort(b)[-10:]]  # top 10 probabilities
        f = b[np.argsort(b)[-20:]]  # top 20 probabilities
        
        l5idx = np.where(np.isin(a,c))[0].tolist()
        l10idx = np.where(np.isin(a,e))[0].tolist()
        l20idx = np.where(np.isin(a,f))[0].tolist()
    
        return_l5idx = real_return[i, l5idx]                    # listing actual returns for chosen stocks
        return_l10idx = real_return[i, l10idx]
        return_l20idx = real_return[i, l20idx]  
        
        return_l5idx = np.array([x for x in return_l5idx if str(x) != 'nan'])      # only buy stock if return not nan
        return_l10idx = np.array([x for x in return_l10idx if str(x) != 'nan'])    
        return_l20idx = np.array([x for x in return_l20idx if str(x) != 'nan'])
        
        if len(return_l5idx) == 0:
            long5.append(long5[i])
        if len(return_l5idx) != 0:
            long5.append(long5[i]/len(return_l5idx) * (len(return_l5idx) + sum(return_l5idx)))
        if len(return_l10idx) == 0:
            long10.append(long10[i])
        if len(return_l10idx) != 0:
            long10.append(long10[i]/len(return_l10idx) * (len(return_l10idx) + sum(return_l10idx)))
        long20.append(long20[i]/len(return_l20idx) * (len(return_l20idx) + sum(return_l20idx)))
    graph.append(long5)
    
    # Computing returns and Sharpe ratios
    ret_long5 = ((long5 / np.roll(long5, 1, axis = 0)) - 1 )[1:]
    tot_ret_long5 = long5[len(long5)-1]/long5[0] - 1
    annual_ret_long5 = (1+tot_ret_long5)**(1/19) - 1
    annual_std_long5 = np.std(ret_long5) * np.sqrt(250)
    annual_sharpe_long5 = (annual_ret_long5-0.0325)/annual_std_long5      # risk-free rate 3.25%
    
    ret_long10 = ((long10 / np.roll(long10, 1, axis = 0)) - 1 )[1:]
    tot_ret_long10 = long10[len(long10)-1]/long10[0] - 1
    annual_ret_long10 = (1+tot_ret_long10)**(1/19) - 1
    annual_std_long10 = np.std(ret_long10) * np.sqrt(250)
    annual_sharpe_long10 = (annual_ret_long10-0.0325)/annual_std_long10   
    
    ret_long20 = ((long20 / np.roll(long20, 1, axis = 0)) - 1 )[1:]
    tot_ret_long20 = long20[len(long20)-1]/long20[0] - 1
    annual_ret_long20 = (1+tot_ret_long20)**(1/19) - 1
    annual_std_long20 = np.std(ret_long20) * np.sqrt(250)
    annual_sharpe_long20 = (annual_ret_long20-0.0325)/annual_std_long20   
    
    OSEBX_return = ((OSEBX / np.roll(OSEBX, 1, axis = 0)) - 1 )[1:]
    tot_ret_OSEBX = OSEBX[len(OSEBX)-1]/OSEBX[0] - 1
    tot_ret_OSEBX = tot_ret_OSEBX[0]
    annual_ret_OSEBX = (1+tot_ret_OSEBX)**(1/19)-1
    annual_std_OSEBX = np.std(OSEBX_return) * np.sqrt(250)
    annual_sharpe_OSEBX = (annual_ret_OSEBX-0.0325)/annual_std_OSEBX      
    
    # defining function to calculate max drawdown
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
    
    mdd_long5 = max_drawdown(long5)
    mdd_long10 = max_drawdown(long10)
    mdd_long20 = max_drawdown(long20)
    mdd_osebx = max_drawdown(OSEBX)[0]
    
    # calculation value at risk 5%
    var95_5 = norm.ppf(1-0.95, annual_ret_long5, annual_std_long5)
    var95_10 = norm.ppf(1-0.95, annual_ret_long10, annual_std_long10)
    var95_20 = norm.ppf(1-0.95, annual_ret_long20, annual_std_long20)
    var95_osebx = norm.ppf(1-0.95, annual_ret_OSEBX, annual_std_OSEBX)
    
    # appending to table
    summary = [annual_ret_long5, annual_std_long5, annual_sharpe_long5, mdd_long5, var95_5, annual_ret_long10, annual_std_long10, annual_sharpe_long10,
               mdd_long10, var95_10, annual_ret_long20, annual_std_long20, annual_sharpe_long20, mdd_long20, var95_20,
               annual_ret_OSEBX, annual_std_OSEBX, annual_sharpe_OSEBX, mdd_osebx, var95_5]
    table.append(summary)

# Adding headers and names
table = pd.DataFrame(table)
table.columns = ['Annual return', 'Std', 'Sharpe ratio', 'Max drawdown', 'VaR 5%', 'Annual return', 'Std', 'Sharpe ratio', 'Max drawdown', 'VaR 5%', 
                 'Annual return', 'Std', 'Sharpe ratio', 'Max drawdown', 'VaR 5%', 'Annual return', 'Std', 'Sharpe ratio', 'Max drawdown', 'VaR 5%']
table.index = ['Logistic', 'RAF', 'SVM', 'LSTM_1', 'LSTM_opt', 'LSTM_d', 'LSTM_f'] 
table = round(table, 4)
table = np.transpose(table)

#table.to_csv('performance table Ks.csv', index=True, header=True, sep=',')

# OSEBX index
osebx_portfolio = [100]
OSEBX_return = ((OSEBX / np.roll(OSEBX, 1, axis = 0)) - 1 )[1:]
for i in range(0, len(OSEBX_return)):
    osebx_portfolio.append(osebx_portfolio[i]*(1+OSEBX_return[i,0]))
osebx_portfolio = osebx_portfolio[:4751]

# Plot cumulative returns before transaction costs
fig, ax = plt.subplots()
line1, = ax.plot(dates, graph[0], color = 'green', label='Logistic')
line2, = ax.plot(dates, graph[1], color = 'magenta', label='RAF')
line3, = ax.plot(dates, graph[2], color = 'red', label='SVM')
line4, = ax.plot(dates, graph[3], color = 'gold', label='LSTM_i')
line5, = ax.plot(dates, graph[4], color = 'royalblue', label='LSTM_a')
line6, = ax.plot(dates, graph[5], color = 'blue', label='LSTM_d')
line7, = ax.plot(dates, graph[6], color = 'maroon', label='LSTM_f')
line7, = ax.plot(dates, osebx_portfolio, color = 'black', label='OSEBX', linewidth=3.5)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yscale('log')
ax.legend()
ax.set_ylabel(r"Portfolio value", labelpad=5)
ax.set_xlabel(r"Time", labelpad=5)
#plt.savefig('performance graph k10 all.png', bbox_inches='tight', dpi=500)
