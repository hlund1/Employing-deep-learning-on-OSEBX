# Directory: Main Github folder containing all folders
import numpy as np
import pandas as pd

#---------------------- Importing files ---------------------
dataset = pd.read_csv('01 Data/01 Input data/query0_vol.csv', encoding='cp1252', sep=';', decimal=',')
returns_order = pd.read_csv('01 Data/02 Preprocessed data/daily prices OSEBX model.csv', sep=';')
constituents = pd.read_csv('01 Data/01 Input data/OSEBX_constituents_ISIN.csv', sep=';')


# Preprocessing the imported files
dataset[['Date', 'Time']] = dataset['TradeDate'].str.split(' ', n = 1, expand = True)
dataset['Date'] = pd.to_datetime(dataset['Date'], format='%d.%m.%Y')
del dataset['Time'],  dataset['TradeDate']
isin_order = returns_order.columns.tolist()
constituents = constituents['ISIN NO']
constituents = constituents.tolist()

# --------------------------------- Prices ---------------------------------
prices = dataset[['Date', 'ISIN', 'AdjGenDividend']]
prices = prices.pivot(index='Date', columns='ISIN', values='AdjGenDividend')
prices = prices[prices.columns.intersection(constituents)]
prices.sort_values(by=['Date'])
prices = prices[isin_order[1:]]

# --------------------------------- Volume ---------------------------------
volume = dataset[['Date', 'ISIN', 'Volume']]
volume = volume.pivot(index='Date', columns='ISIN', values='Volume')
volume = volume[volume.columns.intersection(constituents)]
volume.sort_values(by=['Date'])
volume = volume[isin_order[1:]]

# --------------------------------- Spread ---------------------------------
dataset['AdjMid'] = np.nanmean(dataset[['AdjOffer', 'AdjBid']], axis=1)
dataset['Spread'] = (dataset['AdjOffer']-dataset['AdjBid'])/dataset['AdjMid']
dataset['Spread'] = dataset['Spread']/2 # one-way spread

spread = dataset[['Date', 'ISIN', 'Spread']]
spread = spread.pivot(index='Date', columns='ISIN', values='Spread')
spread = spread[spread.columns.intersection(constituents)]
spread.sort_values(by=['Date'])
spread = spread[isin_order[1:]]

# Fill missing values with average for that stock in different time intervals
for i in range (1,235):
    spread.iloc[0:1500,i].fillna(np.nanmean(spread.iloc[0:1500,i]), inplace=True)
    spread.iloc[1500:3000,i].fillna(np.nanmean(spread.iloc[1500:3000,i]), inplace=True)    
    spread.iloc[3000:5523,i].fillna(np.nanmean(spread.iloc[3000:5523,i]), inplace=True)

# ------------------------------ Spread-adjusted returns ------------------------------ 
adj_offer = dataset[['Date', 'ISIN', 'AdjOffer']]
adj_bid = dataset[['Date', 'ISIN', 'AdjBid']]
adj_offer = adj_offer.pivot(index='Date', columns='ISIN', values='AdjOffer')
adj_bid = adj_bid.pivot(index='Date', columns='ISIN', values='AdjBid')

adj_offer = adj_offer[adj_offer.columns.intersection(constituents)]
adj_offer.sort_values(by=['Date'])
adj_bid = adj_bid[adj_bid.columns.intersection(constituents)]
adj_bid.sort_values(by=['Date'])
adj_offer = adj_offer[isin_order[1:]]
adj_bid = adj_bid[isin_order[1:]]

bid_shifted = np.roll(adj_bid, -1, axis=0)
ask_bid_returns = ((bid_shifted-adj_offer)/adj_offer)
ask_bid_returns = pd.DataFrame(ask_bid_returns)

#--------------------- Saving output data to CSV ---------------------
prices.to_csv('01 Data/02 Preprocessed data/daily prices OSEBX_proc.csv', header = None, sep=',')
volume.to_csv('01 Data/02 Preprocessed data/daily volume OSEBX_proc.csv', header = None, sep=',')
spread.to_csv('01 Data/02 Preprocessed data/daily spread OSEBX_proc.csv', header = None, sep=',')
ask_bid_returns.to_csv('01 Data/02 Preprocessed data/daily bid-ask adj returns OSEBX_proc.csv', sep=',')