# Directory: Main Github folder containing all folders
import numpy as np
import pandas as pd
import pywt

#---------------------- Importing files ---------------------
osebx_daily = pd.read_csv('01 Data/02 Preprocessed data/daily prices OSEBX model.csv', sep=';')
usd_kurs = pd.read_excel('01 Data/01 Input data/valutakurser_norges_bank.xlsx').iloc[5:, [0, 32]]
renter = pd.read_excel('01 Data/01 Input data/renter_daglig_norges_bank.xlsx').iloc[4:, 0:7]
volume = pd.read_csv('01 Data/02 Preprocessed data/daily volume OSEBX_proc.csv', header = None).iloc[:,1:]    
spread = pd.read_csv('01 Data/02 Preprocessed data/daily spread OSEBX_proc.csv', header = None).iloc[:,1:] 
raw_dataset = pd.read_csv('01 Data/02 Preprocessed data/daily prices OSEBX model.csv', sep = ';') 
raw_binary_matrix = pd.read_csv('01 Data/01 Input data/OSEBX_constituents_binary matrix.csv', sep = ';')
spread_returns = pd.read_csv('01 Data/02 Preprocessed data/daily bid-ask adj returns OSEBX_proc.csv', sep=',').iloc[:,1:] 


#---------------------- Creating moving USD/NOK daily ---------------------
# Downloading datasets
usd_kurs = np.flipud(usd_kurs)
usd_kurs = pd.DataFrame(usd_kurs)
usd_kurs = usd_kurs.iloc[3926:,:]

# Date as to right format 
usd_kurs['Date'] = [d.date() for d in usd_kurs[0]]
cols = usd_kurs.columns.tolist()
cols = ['Date', 1]
usd_kurs = usd_kurs[cols]
usd_kurs['Date'] = pd.to_datetime(usd_kurs['Date'], format='%Y-%m-%d')

# Filtering out right observations
usd_kurs = usd_kurs.loc[usd_kurs['Date'].isin(osebx_daily['Date'])==True]

usd_nok = np.empty(shape=(5523, 235))
for i in range(0,235):
    usd_nok[:, i] = usd_kurs.iloc[:, 1]
#---------------------- Creating lange renter ---------------------

renter = np.flipud(renter)
renter = pd.DataFrame(renter)

# Date as to right format 
renter['Date'] = [d.date() for d in renter[0]]
cols = renter.columns.tolist()
cols = ['Date', 0, 1, 2, 3, 4, 5, 6]
renter = renter[cols]
renter['Date'] = pd.to_datetime(renter['Date'], format='%Y-%m-%d')

# Filtering out right observations
renter = renter.loc[renter['Date'].isin(osebx_daily['Date'])==True]

rente_3y = renter.iloc[:, [0, 5]]
rente_10y = renter.iloc[:, [0,7]]

r_3y = np.empty(shape=(5523, 235))          # 3-year government bond yield
r_10y = np.empty(shape=(5523, 235))         # 10-years government bond yield

for i in range(0,235):
    r_3y[:, i] = rente_3y.iloc[:, 1]
    r_10y[:, i] = rente_10y.iloc[:,1]
    
 
#---------------------- Creating moving average ----------------------
dataset = raw_dataset.iloc[:, 1:].values
data = (np.isnan(dataset)*1 - 1) * -1
for i in range(0, data.shape[1]): # Targeted NaN removal
    last_trade = np.where(data[:,i]==1)[0][-1]
    for k in range(1, data.shape[0]):
        if k <= last_trade:
            if np.isnan(dataset[k,i]) == True:
                if np.isnan(dataset[(k+1):(k+100),i]).all() == False: 
                    dataset[k,i] = dataset[k-1,i]
        
ma_50 = np.empty(shape=(5523, 235))         # 50 days moving average
ma_50[:, :] = np.nan

for i in range(0, len(ma_50)):              
    if i==0:
        ma_50[i, :] = dataset[i, :]
    elif i < 50:
        ma_50[i, :] = sum(dataset[0:i, :])/i
    else:
        ma_50[i, :] = sum(dataset[(i-50):i, :])/50
        
        
ma_200 = np.empty(shape=(5523, 235))        # 200 days moving average
ma_200[:, :] = np.nan

for i in range(0, len(ma_200)):             
    if i==0:
        ma_200[i, :] = dataset[i, :]
    elif i < 200:
        ma_200[i, :] = sum(dataset[0:i, :])/i
    else:
        ma_200[i, :] = sum(dataset[(i-200):i, :])/200

#---------------------- Creating simple returns ----------------------
binary_matrix = raw_binary_matrix.iloc[:, 1:].values
dataset_shifted = np.roll(dataset, -1, axis=0)
simple_returns = ((dataset_shifted-dataset)/dataset)

#---------------------- Wavelet transformation ----------------------
# De-noising simple returns
usd_nok = pd.DataFrame(usd_nok)
denoised_returns = np.empty((len(simple_returns), 235))
for i in range(0,235):              
    (ca, cd) = pywt.dwt(simple_returns[:,i], "haar")                
    cat = pywt.threshold(ca, np.nanstd(ca), mode="soft")                
    cdt = pywt.threshold(cd, np.nanstd(cd), mode="soft")                
    denoised_returns[:,i] = pywt.idwt(cat, cdt, "haar")[:-1]
    
# De-noising spread
denoised_spread = np.empty((len(ma_200), 235))
for i in range(0,235):              
    (ca, cd) = pywt.dwt(spread.iloc[:,i], "haar")                
    cat = pywt.threshold(ca, np.nanstd(ca), mode="soft")                
    cdt = pywt.threshold(cd, np.nanstd(cd), mode="soft")                
    denoised_spread[:,i] = pywt.idwt(cat, cdt, "haar")[:-1]

# De-noising volume
denoised_vol = np.empty((len(ma_200), 235))
for i in range(0,235):              
    (ca, cd) = pywt.dwt(volume.iloc[:,i], "haar")                
    cat = pywt.threshold(ca, np.nanstd(ca), mode="soft")                
    cdt = pywt.threshold(cd, np.nanstd(cd), mode="soft")                
    denoised_vol[:,i] = pywt.idwt(cat, cdt, "haar")[:-1]
    
# De-noising usd/nok
denoised_usd_nok = np.empty((len(ma_200), 235))
for i in range(0,235):              
    (ca, cd) = pywt.dwt(usd_nok.iloc[:,i], "haar")                
    cat = pywt.threshold(ca, np.nanstd(ca), mode="soft")                
    cdt = pywt.threshold(cd, np.nanstd(cd), mode="soft")                
    denoised_usd_nok[:,i] = pywt.idwt(cat, cdt, "haar")[:-1]
    
# De-noising spread_returns
denoised_spread_returns = np.empty((len(ma_200), 235))
for i in range(0,235):              
    (ca, cd) = pywt.dwt(spread_returns.iloc[:,i], "haar")                
    cat = pywt.threshold(ca, np.nanstd(ca), mode="soft")                
    cdt = pywt.threshold(cd, np.nanstd(cd), mode="soft")                
    denoised_spread_returns[:,i] = pywt.idwt(cat, cdt, "haar")[:-1]


#--------------------- Saving output data to CSV ---------------------


#--------------------- LSTM_f input data: saving to CSV ---------------------
input_data = [simple_returns, spread, volume, usd_nok, ma_50, ma_200]
input_data = np.dstack(input_data)                      # Converting 2D to 3D ndarray
np.save('01 Data/02 Preprocessed data/input data LSTM_f 6 features', input_data)

#--------------------- LSTM_Var3 input data: saving to CSV ---------------------
input_data = [spread_returns, spread, volume]
input_data = np.dstack(input_data)                      # Converting 2D to 3D ndarray
np.save('01 Data/02 Preprocessed data/input data LSTM_Var3', input_data)

#--------------------- LSTM_Var3D input data: saving to CSV ---------------------
input_data = [denoised_spread_returns, denoised_spread, denoised_vol]
input_data = np.dstack(input_data)                      # Converting 2D to 3D ndarray
np.save('01 Data/02 Preprocessed data/input data LSTM_Var3D', input_data)

#--------------------- LSTM_Var8D input data: saving to CSV ---------------------
rente_3y, rente_10y = np.array(rente_3y), np.array(rente_10y)
input_data = [denoised_spread_returns, denoised_spread, denoised_vol, ma_50, ma_200, denoised_usd_nok, r_3y, r_10y]
input_data = np.dstack(input_data)                      # Converting 2D to 3D ndarray
np.save('01 Data/02 Preprocessed data/input data LSTM_Var8D', input_data)
