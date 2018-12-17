# %reset -f
#----------------------------- Importing packages ----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os
from scipy import stats
from math import sqrt

#----------------------------- Importing data ----------------------------
OSEBX = pd.read_excel('01 Data/01 Input data/OSEBX_index 1996-2018.xlsx').iloc[190:4962, 1:2].values
real_return = pd.read_csv('01 Data/02 Preprocessed data/daily real return OSEBX_proc.csv', header = None)
spreads = pd.read_csv('01 Data/02 Preprocessed data/daily spread OSEBX_proc.csv', header = None)
volume = pd.read_csv('01 Data/02 Preprocessed data/daily volume OSEBX_proc.csv', header = None)
dates = pd.read_csv('01 Data/02 Preprocessed data/daily prices OSEBX_proc.csv', sep = ',', header = None).iloc[772:5523, 0] 
OSEBX = np.flipud(OSEBX)
real_return = np.array(real_return)
spreads = spreads.iloc[:,1:]
volume = volume.iloc[:,1:]
spreads, volume = spreads.astype('float64'), volume.astype('float64')
dates = pd.to_datetime(dates, format="%Y-%m-%d")

for i in range(0, spreads.shape[1]):
        spreads.iloc[:,i].fillna(np.nanmean(spreads.iloc[:,i]), inplace = True)
spreads, volume = np.array(spreads), np.array(volume)
spreads = np.nan_to_num(spreads)

# Retrieving files for probability and prediction
files_prob = sorted(glob.glob(os.path.join('04 Predictions/02 Chapter 6 Practitioner/01 prob/', "*.csv"))) 
files_class = sorted(glob.glob(os.path.join('04 Predictions/02 Chapter 6 Practitioner/02 class/', "*.csv")))  

LSTM_Standard = pd.read_csv(files_prob[1], header = None) 
LSTM_Spread = pd.read_csv(files_prob[0], header = None) 
LSTM_Var3 = pd.read_csv(files_prob[2], header = None)
LSTM_Var3D = pd.read_csv(files_prob[3], header = None) 
LSTM_Var8D = pd.read_csv(files_prob[4], header = None)

osebx_portfolio = [100]
OSEBX_return = ((OSEBX / np.roll(OSEBX, 1, axis = 0)) - 1 )[1:]
for i in range(0, len(OSEBX_return)):
    osebx_portfolio.append(osebx_portfolio[i]*(1+OSEBX_return[i,0]))
osebx_portfolio = osebx_portfolio[:4751]

#----------------------------- Defining functions ----------------------------

# Calculates portfolio value after transaction costs and bid-ask spread
def aftertxspread(prob, k):
    tx = 0.00029
    #tx = 0.001555 break even tx
    prob = np.array(prob)
    long10 = [100]
    l10idxx = []
    weight = []
    for i in range(0, len(prob)):
        a = prob[i,]
        b = a[~np.isnan(a)]
        e = b[np.argsort(b)[-k:]] # top k probabilities
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
    for t in range(0, len(prob)):
        buy = [] # at t=0, wether to buy
        sell = [] # at t = 1, whether to sell
        
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
            temp = sum((long10[t]*np.array(weight[t]))*(1+(real_return[t,l10idxx[t]]-(tx+spreads[t,l10idxx[t]])*power)))
        long10.append(temp)
    return long10

# Calculates portfolio value after transaction costs and bid-ask spread, while requiring a certain trading volume of stocks
def restricted_volume(prob, k = 10, pctile = 25): 
    quartilevolume = np.nanpercentile(volume, pctile, axis = 1)
    tx = 0.00029
    #tx = 0.001555 break even tx
    prob = np.array(prob)
    long10 = [100]
    l10idxx = []
    weight = []
    for i in range(0, len(prob)):
        a = prob[i,]
        b = a[~np.isnan(a)]
        e = b[np.argsort(b)[-k:]] # top k probabilities
        l10idx = np.where(np.isin(a,e))[0].tolist()
        l10idxx.append(l10idx)  
        
        del_list = []
        for s in range(0, len(l10idxx[i])):
            if str(real_return[i, l10idxx[i][s]]) == 'nan':
                del_list.append(l10idxx[i][s])
        
        if len(del_list) != 0:
            for x in range(0, len(del_list)):
                l10idxx[i][:] = (value for value in l10idxx[i] if value != del_list[x])
        
        te = []
        for u in l10idxx[i]:
            if np.isnan(np.nanmean(volume[(i-10):i,u])) == False:
                if np.nanmean(volume[(i-10):i,u]) >= quartilevolume[i]:
                    te.append(u)
            if np.isnan(np.nanmean(volume[(i-10):i,u])):
                te.append(u)
        #np.nanmean(volume[(i-10):i,u])
        l10idxx[i] = te
        
        if len(l10idxx[i][:]) == 0:
            weight.append(0)
        if len(l10idxx[i][:]) != 0:
            weights_t=[1/len(l10idxx[i])]*len(l10idxx[i])
            weight.append(weights_t)
        
    for t in range(0, len(prob)):
        buy = [] # at t=0, wether to buy
        sell = [] # at t = 1, whether to sell
        
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
            temp = sum((long10[t]*np.array(weight[t]))*(1+(real_return[t,l10idxx[t]]-(tx+spreads[t,l10idxx[t]])*power)))
        long10.append(temp)
    return long10

# Calculates portfolio value after transaction costs and bid-ask spread, while setting a threshold for the probability of the selected stocks
def restricted_treshold(prob, k = 10, treshold = 0.6): # restricts to a certain level of confidence
    tx = 0.00029
    prob = np.array(prob)
    long10 = [100]
    l10idxx = []
    weight = []
    for i in range(0, len(prob)):
        a = prob[i,]
        b = a[~np.isnan(a)]
        e = b[np.argsort(b)[-k:]] # top k probabilities
        e = e[e>=treshold]
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
        
    for t in range(0, len(prob)):
        buy = [] # at t=0, wether to buy
        sell = [] # at t = 1, whether to sell
        
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
            temp = sum((long10[t]*np.array(weight[t]))*(1+(real_return[t,l10idxx[t]]-(tx+spreads[t,l10idxx[t]])*power)))
        long10.append(temp)
    return long10

# Calculates portfolio value after transaction costs and bid-ask spread, while limiting the turnover of the portfolio

def restricted_turnover(prob, k, multiplier = 2, multiplier2 = 2): 
    tx = 0.00029
    prob = np.array(prob)
    long10 = [100]
    l10idxx = []
    weight = []
    for i in range(0, len(prob)):
        a = prob[i,]
        b = a[~np.isnan(a)]
        e = b[np.argsort(b)[-k:]] # top k probabilities
        f = b[np.argsort(b)[-int(k*multiplier):]] # top k*2 probabilities
        l10idx = np.where(np.isin(a,e))[0].tolist()
        
        midl = []
        if i == 0:
            l10idxx.append(l10idx)  
        if i != 0:
            l10idx_ = l10idxx[i-1]
            for u in l10idx_:
                if a[u] in f:
                    midl.append(u)
            for z in range(1,int(k*multiplier2)):
                if len(midl) < k:
                    update = np.where(np.isin(a,b[np.argsort(b)[-int(k*multiplier2):]]))[0][z]
                    if update not in midl:
                        midl.append(update)
            l10idxx.append(midl)  
        
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
        
    for t in range(0, len(prob)):
        buy = [] # at t=0, wether to buy
        sell = [] # at t = 1, whether to sell
        
        if t == 0:
            for s in l10idxx[t]:
                if s not in l10idxx[t+1]:
                    sell.append(s)
                buy.append(s)
    
        if (t>0) and (t<4749):
            for s in l10idxx[t]:
                if s not in l10idxx[t+1]:
                    sell.append(s)
            for n in l10idxx[t]:
                if n not in l10idxx[t-1]:
                    buy.append(n)
        
        if t == 4749:
            for n in l10idxx[t]:
                if n not in l10idxx[t-1]:
                    buy.append(n)
                sell.append(n)
        
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
            temp = sum((long10[t]*np.array(weight[t]))*(1+(real_return[t,l10idxx[t]]-(tx+spreads[t,l10idxx[t]])*power)))
        long10.append(temp)
    return long10

# Calculates portfolio value after transaction costs and bid-ask spread, while only rebalancing portfolio every fifth day
def restricted_5day(prob, k):
    tx = 0.00029
    #tx = 0.001555 break even tx
    prob = np.array(prob)
    long10 = [100]
    l10idxx = []
    weight = []
    for i in range(0, len(prob)):
        if (i == 0) or ((i % 5) == 0):
            a = prob[i,]
            b = a[~np.isnan(a)]
            e = b[np.argsort(b)[-k:]] # top k probabilities
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
        if ((i % 5) != 0) and (i != 0):
            l10idxx.append(l10idxx[i-1])
            weight.append(weight[i-1])
            
    for t in range(0, len(prob)):
        buy = [] # at t=0, wether to buy
        sell = [] # at t = 1, whether to sell
        
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
            real_return[t,l10idxx[t]] = np.nan_to_num(real_return[t,l10idxx[t]])
            temp = sum((long10[t]*np.array(weight[t]))*(1+(real_return[t,l10idxx[t]]-(tx+spreads[t,l10idxx[t]])*power)))
        long10.append(temp)
    return long10

# Calculates portfolio value after transaction costs and bid-ask spread, while implementing bet sizing
def restricted_bz(prob, k):
    tx = 0.00029
    #tx = 0.001555 break even tx
    prob = np.array(prob)
    long10 = [100]
    l10idxx = []
    weight = []
    for i in range(0, len(prob)):
        a = prob[i,]
        b = a[~np.isnan(a)]
        e = b[np.argsort(b)[-k:]] # top k probabilities
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
            avg = sum(b)/len(b)
            stdev = np.std(b)
            f = a[l10idxx[i]]
            weights_t = (f-avg) / stdev
            weights_t = weights_t / sum(weights_t)
            weight.append(weights_t)
        
    for t in range(0, len(prob)):
        buy = [] # at t=0, wether to buy
        sell = [] # at t = 1, whether to sell
        
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
            temp = sum((long10[t]*np.array(weight[t]))*(1+(real_return[t,l10idxx[t]]-(tx+spreads[t,l10idxx[t]])*power)))
        long10.append(temp)
    return long10

# Calculates portfolio value after transaction costs and bid-ask spread, while setting a maximum bid-ask spread level of stocks
def restricted_spreads(prob, k, maxspread = 0.01):
    tx = 0.00029
    #tx = 0.001555 break even tx
    prob = np.array(prob)
    long10 = [100]
    l10idxx = []
    weight = []
    for i in range(0, len(prob)):
        a = prob[i,]
        b = a[~np.isnan(a)]
        e = b[np.argsort(b)[-k:]] # top k probabilities
        l10idx = np.where(np.isin(a,e))[0].tolist()
        l10idxx.append(l10idx)  
        
        del_list = []
        for y in l10idxx[i]:
            if np.isnan(np.nanmean(spreads[(i-10):i,y])) == False:
                if np.nanmean(spreads[(i-10):i,y]) >= maxspread:
                    del_list.append(y)
                
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
        
    for t in range(0, len(prob)):
        buy = [] # at t=0, wether to buy
        sell = [] # at t = 1, whether to sell
        
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
            temp = sum((long10[t]*np.array(weight[t]))*(1+(real_return[t,l10idxx[t]]-(tx+spreads[t,l10idxx[t]])*power)))
        long10.append(temp)
    return long10

# Same as the restricted turnover above, only that the output is the portfolio composition and not the portfolio value
def restricted_turnover_idx(prob, k, multiplier = 2): # limits sell-offs: Selger bare hvis ikke blant topp k*multiplier
    tx = 0.00029
    prob = np.array(prob)
    long10 = [100]
    l10idxx = []
    weight = []
    for i in range(0, len(prob)):
        a = prob[i,]
        b = a[~np.isnan(a)]
        e = b[np.argsort(b)[-k:]] # top k probabilities
        f = b[np.argsort(b)[-int(k*multiplier):]] # top k*2 probabilities
        l10idx = np.where(np.isin(a,e))[0].tolist()
        
        midl = []
        if i == 0:
            l10idxx.append(l10idx)  
        if i != 0:
            l10idx_ = l10idxx[i-1]
            for u in l10idx_:
                if a[u] in f:
                    midl.append(u)
            for z in range(1,(k*2)):
                if len(midl) < k:
                    update = np.where(np.isin(a,b[np.argsort(b)[-(k*2):]]))[0][z]
                    if update not in midl:
                        midl.append(update)
            l10idxx.append(midl)  
        
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
        
    for t in range(0, len(prob)):
        buy = [] # at t=0, wether to buy
        sell = [] # at t = 1, whether to sell
        
        if t == 0:
            for s in l10idxx[t]:
                if s not in l10idxx[t+1]:
                    sell.append(s)
                buy.append(s)
    
        if (t>0) and (t<4749):
            for s in l10idxx[t]:
                if s not in l10idxx[t+1]:
                    sell.append(s)
            for n in l10idxx[t]:
                if n not in l10idxx[t-1]:
                    buy.append(n)
        
        if t == 4749:
            for n in l10idxx[t]:
                if n not in l10idxx[t-1]:
                    buy.append(n)
                sell.append(n)
        
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
            temp = sum((long10[t]*np.array(weight[t]))*(1+(real_return[t,l10idxx[t]]-(tx+spreads[t,l10idxx[t]])*power)))
        long10.append(temp)
    return l10idxx

# Calculates return statistics of the portfolios

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

# Output 1.1 - Left
fig, ax = plt.subplots()
line1, = ax.plot(dates, aftertxspread(LSTM_Standard, k = 10), color = 'black', label='LSTM_Standard')
line2, = ax.plot(dates, aftertxspread(LSTM_Spread, k = 10), color = 'royalblue', label='LSTM_Spread')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.ylabel('Log cumulative return')
plt.yscale('log')
ax.legend()
ax.set_ylabel(r"Log cumulative return", labelpad=5)
ax.set_xlabel(r"Time", labelpad=5)
#plt.savefig('Output/training_left.png', bbox_inches='tight')

# Output 1.2 - Right
fig, ax = plt.subplots()
line1, = ax.plot(dates, aftertxspread(LSTM_Spread, k = 10), color = 'royalblue', label='LSTM_Spread')
line2, = ax.plot(dates, aftertxspread(LSTM_Var3, k = 10), color = 'red', label='3_Features')
line3, = ax.plot(dates, aftertxspread(LSTM_Var3D, k = 10), color = 'maroon', label='3_Features_d')
line5, = ax.plot(dates, aftertxspread(LSTM_Var8D, k = 10), color = 'green', label='8_Features_d')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.ylabel('Log cumulative return')
plt.yscale('log')
ax.legend()
ax.set_ylabel(r"Log cumulative return", labelpad=5)
ax.set_xlabel(r"Time", labelpad=5)
#plt.savefig('Output/training_right.png', bbox_inches='tight')

# Output 3
fig, ax = plt.subplots()
line1, = ax.plot(dates, aftertxspread(LSTM_Var8D, k = 5), color = 'black', label='Standard')
line2, = ax.plot(dates, restricted_volume(LSTM_Var8D, k = 5, pctile = 5), color = 'b', label='Volume')
line3, = ax.plot(dates, restricted_turnover(LSTM_Var8D, k = 5, multiplier = 3), color = 'red', label='Turnover')
line4, = ax.plot(dates, restricted_5day(LSTM_Var8D, k = 5), color = 'g', label='Weekly')
line5, = ax.plot(dates, restricted_bz(LSTM_Var8D, k = 5), color = 'y', label='Bet sizing')
line6, = ax.plot(dates, restricted_treshold(LSTM_Var8D, k = 5, treshold = 0.5375), color = 'm', label='Threshold')
line7, = ax.plot(dates, restricted_spreads(LSTM_Var8D, k = 5, maxspread = 0.060), color = 'c', label='Spread')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.ylabel('Log cumulative return')
plt.yscale('log')
ax.legend()
ax.set_ylabel(r"Log cumulative return", labelpad=5)
ax.set_xlabel(r"Time", labelpad=5)
#plt.savefig('Output/different_strategies.png', bbox_inches='tight', dpi = 500)

# Output 4
fig, ax = plt.subplots()
line3, = ax.plot(dates, restricted_turnover(LSTM_Var8D, k = 5, multiplier = 3), color = 'royalblue', label='Turnover')
line4, = ax.plot(dates, osebx_portfolio, color = 'black', label='OSEBX')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yscale('log')
ax.legend(loc = 2)
ax.set_ylabel(r"Portfolio value", labelpad=5)
ax.set_xlabel(r"Time", labelpad=5)
#plt.savefig('Output/bestvsosebx.png', bbox_inches='tight', dpi = 500)

# Statistical significance of performance vs OSEBX - Probabalistic Sharpe Ratio
best = restricted_turnover(LSTM_Var8D, k = 5, multiplier = 3)
# Daily returns
ret_osebx = ((osebx_portfolio / np.roll(osebx_portfolio, 1, axis = 0)) - 1 )[1:]
ret_best = ((best / np.roll(best, 1, axis = 0)) - 1 )[1:]
daily_ret_osebx = (osebx_portfolio[len(osebx_portfolio)-1]/osebx_portfolio[0])**(1/4750) - 1
daily_ret_best = (best[len(osebx_portfolio)-1]/best[0])**(1/4750) - 1

# Daily stdev
std_osebx = np.std(ret_osebx)
std_best = np.std(ret_best)
# Daily sharpe
sp_osebx = daily_ret_osebx / std_osebx # 0.02592
sp_best = daily_ret_best / std_best

# Skewness & Kurtosis
skew = stats.skew(ret_best)
kurt = stats.kurtosis(ret_best)

# Probabalistic sharpe ratio calculation
z = ((sp_best-sp_osebx)*sqrt(4750-1)) / sqrt(1 - skew*sp_osebx + ((kurt-1)/4)*(sp_best**2))
pvalue = 1 - stats.norm.cdf(z)


# Save portfolio composition of best model/strategy
lidxx = restricted_turnover_idx(LSTM_Var8D, k = 5, multiplier = 3)
lidxx = np.array(lidxx)
np.save('01 Data/02 Preprocessed data/Portfolio composition after spreads.npy', lidxx)
