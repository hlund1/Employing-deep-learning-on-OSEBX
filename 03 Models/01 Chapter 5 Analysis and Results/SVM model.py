# %reset -f
#----------------------------- Importing packages ----------------------------
import numpy as np
import pandas as pd
from sklearn import svm

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
# Bruker her klassifisering for over og under medianen
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

#Making empty output dataframes
predicted_returns_class, predicted_returns_prob = np.empty(shape=(study_periods*test,dataset.shape[1])), np.empty(shape=(study_periods*test,dataset.shape[1]))
predicted_returns_class[:,:], predicted_returns_prob[:,:] = np.nan, np.nan
lookback = 240

#Defining classifier:
classifier = svm.SVC(C = 100, kernel = 'rbf', gamma = 0.01, probability = True)

#------------ This is where major loop over all 19 periods starts -------------
for i in range(0, len(return_window)):
    # Determine which stocks are eligleble for the study period
    
    vec0 = list(list(np.where(binary_matrix[(749+i*test), :]==1))[0])
    vec = [] 
    for u in vec0: # Denne skal ikke være nødvendig
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
   
    # Making the predictions for each period:
    inputs1 = return_window[i][days-test-lookback:, vec] # Update
    inputs = np.nan_to_num(inputs1, copy = True)    # Update
    inputs = (inputs - mu) / std
    X_test = []
    for k in range(lookback, lookback+test):
        X_test.append(inputs[(k-lookback):k, :])
    X_test = np.array(X_test)

    #------------------------ Looping over all stocks -------------------------     
    for s in range (0, X_train.shape[2]):
        X_train_s = np.reshape(X_train[:, :, s], ((X_train.shape[0], X_train.shape[1])))
        X_test_s = np.reshape(X_test[:, :, s], ((X_test.shape[0], X_test.shape[1])))
        X_test_s = np.nan_to_num(X_test_s, copy = True)
        y_train_s = y_train[:, s]
        classifier.fit(X_train_s, y_train_s) 
        y_pred = classifier.predict(X_test_s)
        y_pred_proba = classifier.predict_proba(X_test_s) 
        y_pred_proba = y_pred_proba[:, 1]
        predicted_returns_class[(test*i):(test+test*i), vec[s]] = y_pred[:] 
        predicted_returns_prob[(test*i):(test+test*i), vec[s]] = y_pred_proba[:]
    #-------------------------------- END LOOP --------------------------------

# Save predictions to CSV
np.savetxt('04 Predictions/01 Chapter 5 Analysis and Results/01 prob/3 svm prob.csv.csv', predicted_returns_prob, delimiter=",")
np.savetxt('04 Predictions/01 Chapter 5 Analysis and Results/01 class/3 svm class.csv.csv', predicted_returns_class, delimiter=",")