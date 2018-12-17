# %reset -f
#----------------------------- Importing packages ----------------------------
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputLayer
from keras.layers import CuDNNLSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping

#----------------------------- Importing the data -----------------------------
raw_dataset = pd.read_csv('01 Data/02 Preprocessed data/daily prices OSEBX model.csv', sep = ';') 
dataset = raw_dataset.iloc[:, 1:].values
data = (np.isnan(dataset)*1 - 1) * -1
for i in range(0, data.shape[1]): # Targeted NaN removal
    last_trade = np.where(data[:,i]==1)[0][-1]
    for k in range(1, data.shape[0]):
        if k <= last_trade:
            if np.isnan(dataset[k,i]) == True:
                if np.isnan(dataset[(k+1):(k+100),i]).all() == False: 
                    dataset[k,i] = dataset[k-1,i]
raw_binary_matrix = pd.read_csv('01 Data/01 Input data/OSEBX_constituents_binary matrix.csv', sep = ';')
binary_matrix = raw_binary_matrix.iloc[:, 1:].values
dataset_shifted = np.roll(dataset, -1, axis=0)
simple_returns = ((dataset_shifted-dataset)/dataset)

#---------------------- Binary matrix for classification ----------------------
median_returns = []
for i in range(0, simple_returns.shape[0]):
    vec = list(list(np.where(binary_matrix[(i), :]==1))[0])
    median_returns.append(np.nanmedian(simple_returns[i,vec], axis=0))
median_returns = np.array(median_returns)
median_returns = np.reshape(median_returns, (-1,1))
binary_returns = (simple_returns >= median_returns)*1

#---------------------- Create study periods of 1000 days ---------------------
return_window = []
return_window_y = []
days = 1000
train, test = int(0.75*days), int(0.25*days)
lookback = 240 
study_periods = int((len(dataset)-train)/test)
for i in range(0, study_periods):
    return_window.append(simple_returns[(test*i):(days+test*i)])
    return_window_y.append(binary_returns[(test*i):(days+test*i)])
return_window, return_window_y = np.array(return_window), np.array(return_window_y)

training_set = []
training_set_y = []
test_set = []
for i in range(0,study_periods):
    training_set.append(return_window[i, :train])
    training_set_y.append(return_window_y[i, :train])
    test_set.append(return_window_y[i, train:])
training_set, test_set, training_set_y = np.array(training_set), np.array(test_set), np.array(training_set_y)

# Preparing empty matrices to fill predictions inn
predicted_returns_prob, predicted_returns_class = np.empty(shape=(study_periods*test,dataset.shape[1])), np.empty(shape=(study_periods*test,dataset.shape[1]))
predicted_returns_prob[:,:], predicted_returns_class[:,:] = np.nan, np.nan

#------------------------ Building the LSTM-model -------------------------
output_units = 1
dp = 0.1 
act = 'sigmoid' 
opt = 'RMSprop' 
lf = 'binary_crossentropy'
ep = 1000 
bs = 32 
val_split = 0.2 
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

# Construction layers:
classifier = Sequential()
classifier.add(InputLayer(input_shape=(lookback, 1)))
classifier.add(CuDNNLSTM(units = 30, return_sequences = True))
classifier.add(Dropout(dp))
classifier.add(CuDNNLSTM(units = 20, return_sequences = False))
classifier.add(Dropout(dp))
classifier.add(Dense(units = output_units, activation = act))
classifier.compile(optimizer = opt, loss = lf)
#------------ Loop over 19 study periods -------------
for i in range(0, len(return_window)):
    
    # Determine which stocks are eligleble for the study period    
    vec0 = list(list(np.where(binary_matrix[(749+i*test), :]==1))[0])
    vec = [] 
    for u in vec0: 
        if (all(np.isnan(return_window[i,0:750,u])) == False and all(np.isnan(return_window[i,750:1000,u])) == False) == True:
            vec.append(u)
    
    # Training set
    training_set_temp1 = training_set[i][:,vec]
    training_set_temp = np.nan_to_num(training_set_temp1, copy = True)
    mu = np.nanmean(training_set_temp)
    std = np.nanstd(training_set_temp)
    training_set_scaled = (training_set_temp - mu) / std
    
    X_train = []
    for j in range(lookback, len(training_set_scaled)):
        X_train.append(training_set_scaled[(j-lookback):j, :])  
    y_train = training_set_y[i][lookback:, vec]
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Test set
    inputs1 = return_window[i][days-test-lookback:, vec] 
    inputs = np.nan_to_num(inputs1, copy = True)
    inputs = (inputs - mu) / std
    X_test = []
    for k in range(lookback, lookback+test):
        X_test.append(inputs[(k-lookback):k, :])
    X_test = np.array(X_test)
    
    #------------ Getting the data on long format -------------
    X_train1 = np.transpose(X_train[0,:,:])
    for x in range(1, 510):
        X_train1 = np.concatenate((X_train1, np.transpose(X_train[x,:,:])))
    y_train1 = np.reshape(y_train, (-1,1)) 
    
    X_test1 = np.transpose(X_test[0,:,:])
    for x in range(1, 250):
        X_test1 = np.concatenate((X_test1, np.transpose(X_test[x,:,:])))
    
    X_train1 = np.reshape(X_train1, ((510*len(vec)),lookback,1))
    X_test1 = np.reshape(X_test1, ((250*len(vec)),lookback,1)) 

    #------------------------ Fitting to all stocks -------------------------
    classifier.fit(X_train1, y_train1, epochs = ep, batch_size = bs, validation_split = val_split, callbacks = [es], shuffle = False)
    pred_prob = classifier.predict(X_test1)
    pred_class = classifier.predict_classes(X_test1)
    
    pred_prob1 = np.reshape(pred_prob, (250,len(vec))) # Back to wide format
    pred_class1 = np.reshape(pred_class, (250,len(vec)))
    predicted_returns_prob[(test*i):(test+test*i), vec] = pred_prob1
    predicted_returns_class[(test*i):(test+test*i), vec] = pred_class1

# Save predictions to CSV
np.savetxt('04 Predictions/01 Chapter 5 Analysis and Results/01 prob/6 lstm_d prob.csv.csv', predicted_returns_prob, delimiter=",")
np.savetxt('04 Predictions/01 Chapter 5 Analysis and Results/01 class/6 lstm_d class.csv.csv', predicted_returns_class, delimiter=",")