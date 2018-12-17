# %reset -f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------ Importing data ------------------------------
# Importing real_returns, binary_matrix
raw_dataset = pd.read_csv('01 Data/02 Preprocessed data/daily prices OSEBX model.csv', sep = ';') 
raw_binary_matrix = pd.read_csv('01 Data/01 Input data/OSEBX_constituents_binary matrix.csv', sep = ';')
OSEBX = pd.read_excel('01 Data/01 Input data/OSEBX_index 1996-2018.xlsx').iloc[190:4962, 1:2].values
real_return = pd.read_csv('01 Data/02 Preprocessed data/daily real return OSEBX_proc.csv', header = None)
binary_returns = pd.read_csv('01 Data/02 Preprocessed data/daily binary return OSEBX_proc.csv', header = None) # 1 if above cross-sectional median return, else 0
identifier = pd.read_csv('01 Data/01 Input data/constituents isin and ticker.csv', header = 0, names = ['1','2','3'])
volume = pd.read_csv('01 Data/02 Preprocessed data/daily volume OSEBX_proc.csv', header = None)
real_return = np.array(real_return)
real_return, binary_returns = np.array(real_return), np.array(binary_returns)
OSEBX = np.flipud(OSEBX)

dataset = raw_dataset.iloc[:, 1:].values
data = (np.isnan(dataset)*1 - 1) * -1
for i in range(0, data.shape[1]): # Targeted NaN removal
    last_trade = np.where(data[:,i]==1)[0][-1]
    for k in range(1, data.shape[0]):
        if k <= last_trade:
            if np.isnan(dataset[k,i]) == True:
                if np.isnan(dataset[(k+1):(k+100),i]).all() == False: 
                    dataset[k,i] = dataset[k-1,i]
binary_matrix = raw_binary_matrix.iloc[:, 1:].values
dataset_shifted = np.roll(dataset, -1, axis=0)
simple_returns = ((dataset_shifted-dataset)/dataset)

# -------------------------------- Analysis I --------------------------------    
lidx6 = np.load('01 Data/02 Preprocessed data/Portfolio composition after spreads.npy')
lidxx = []
for i in range(0,len(lidx6)):
    lidxx.append(lidx6[i])

# counting trades            
frequency = np.empty(shape = (1,235))
frequency[:,:] = 0
for i in range(0,len(lidxx)):
    frequency[:,lidxx[i]] += 1

trades = [len(lidxx[0])]
for i in range(1,len(lidxx)):
    tradetemp = 0
    for v in lidxx[i]:
        if v not in lidxx[i-1]:
            tradetemp += 1
    trades.append(tradetemp)

total_trades = sum(trades)
avg_trades_day = total_trades / len(lidxx)

# importing ticker and isin for all constituents
frequency = np.transpose(frequency)
frequency = pd.DataFrame(frequency)
frequency.columns = ['trades']
data = pd.concat([identifier, frequency], axis = 1)
list(data)
sorted_data = data.sort_values(by = ['trades'])

# Descriptive stats
sum(i > 0 for i in sorted_data['trades'])
most_traded = sorted_data.nlargest(20, 'trades') 



# -------------------------------- Analysis II --------------------------------
# Creating empty table 
desc = np.empty(shape=(len(real_return)-240,241, 2))
desc[:,:,:] = np.nan

for i in range(240, len(real_return)):
    vec = list(list(np.where(binary_matrix[i, :]==1))[0])
    
    # Stocks selected
    l10idx = lidxx[i]
    
    # Cumulative return market
    mkt = [1]
    return_market = real_return[i-240:i,vec]
    return_market_avg = np.nanmean(return_market, axis = 1)
    cum_return = (1 + return_market_avg)
    for m in range(0,240):
        mkt.append(mkt[m]*cum_return[m])
    mkt = np.array(mkt) - 1
    desc[i-240,:,0] = mkt
    
    # Cumulative return K = 10 portfolio
    portfolio = [1]
    return_portfolio = real_return[i-240:i,l10idx]
    return_portfolio_avg = np.nanmean(return_portfolio, axis = 1)
    cum_return_p = (1 + return_portfolio_avg)
    for p in range(0,240):
        portfolio.append(portfolio[p]*cum_return_p[p])
    portfolio = np.array(portfolio) - 1
    desc[i-240,:,1] = portfolio

# Average OSEBX cumreturn last 240 days for all 4510 days
average_cumreturn_osebx = np.nanmean(desc[:,:,0], axis = 0)
average_std_osebx = np.nanstd(desc[:,:,0], axis = 0)
std_osebx = np.mean(average_std_osebx)

# Average Portfolio cumreturn last 240 days for all 4510 days
average_cumreturn_portfolio = np.nanmean(desc[:,:,1], axis = 0)
average_std_portfolio = np.nanstd(desc[:,:,1], axis = 0)
std_portfolio = np.mean(average_std_portfolio)

# Looking at average return, standard deviation and beta for period 240 average K10
beta = np.ma.cov(average_cumreturn_portfolio, average_cumreturn_osebx)[0,1]/np.std(average_cumreturn_osebx)**2

# creating graph for average 240 days
fig, ax = plt.subplots()
line5, = ax.plot(average_cumreturn_portfolio, color = 'royalblue', label='LSTM portfolio')
line7, = ax.plot(average_cumreturn_osebx, color = 'black', label='OSEBX')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.legend()
ax.set_ylabel(r"Cumulative return", labelpad=5)
ax.set_xlabel(r"days", labelpad=5)
#plt.savefig('last 240 days portfolio after spreads.png', bbox_inches='tight', dpi=1000)


'''# analysing volume for top 20 stocks
top_isin = most_traded['1'].tolist()
# median volume for OSEBX
vol_all_mean=[]
for i in range(0,235):
    vol_all_mean.append(volume.iloc[4500:4750, i].mean())
np.nanmedian(vol_all_mean)                  

# median volume for portfolio    
volume_top = volume[volume.columns.intersection(top_isin)]
vol_top_mean=[]
for i in range(0,20):
    vol_top_mean.append(volume_top.iloc[4500:4750, i].mean())
np.nanmedian(vol_top_mean)'''
