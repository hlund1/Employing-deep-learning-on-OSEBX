# %reset -f
#----------------------------- Importing packages ----------------------------
import numpy as np
import pandas as pd
from math import log
import glob, os

real_return = pd.read_csv('01 Data/02 Preprocessed data/daily real return OSEBX_proc.csv', header = None)
binary_returns = pd.read_csv('01 Data/02 Preprocessed data/daily binary return OSEBX_proc.csv', header = None) 

real_return, binary_returns = np.array(real_return), np.array(binary_returns)

# Making list of all files for probability and prediction
files_prob = sorted(glob.glob(os.path.join('04 Predictions/01 Chapter 5 Analysis and Results/01 prob', "*.csv")))       
files_class = sorted(glob.glob(os.path.join('04 Predictions/01 Chapter 5 Analysis and Results/02 class', "*.csv")))

# Making empty table to append each result in
table = []
for x in range(0, len(files_prob)):
    pred_prob = pd.read_csv(files_prob[x], header = None)
    pred_class = pd.read_csv(files_class[x], header = None)

    # Making numpy array
    pred_prob, pred_class, real_return, binary_returns = np.array(pred_prob), np.array(pred_class), np.array(real_return), np.array(binary_returns)
    
    # 1. Directional accuracy
    a = (binary_returns == pred_class)
    b = sum(sum(a))                                     # Number of correct predictions
    tot_pred = sum(sum(~np.isnan(pred_class)))          # number of total predictions
    accuracy = b / tot_pred 
    
    # 2. Confusion Matrix
    pos_pred = np.count_nonzero(pred_class == 1)        # positive predictions
    up_pred_share = pos_pred/tot_pred                   # up predictions
    pred_up = (pred_class == 1)*1.0
    pred_up[pred_up == 0] = np.nan
    real_up = (binary_returns == 1)*1.0
    real_up[real_up == 0] = np.nan
    true_pos = np.count_nonzero(pred_up==real_up)       # True positive predictions
    false_pos = pos_pred - true_pos
    
    neg_pred = np.count_nonzero(pred_class == 0)        # negative predictions
    neg_pred_share = neg_pred/tot_pred                  # down predictions
    pred_zero = (pred_class == 0)*1.0
    pred_zero[pred_zero == 0] = np.nan
    real_zero = (binary_returns == 0)*1.0
    real_zero[real_zero == 0] = np.nan
    true_neg = np.count_nonzero(pred_zero==real_zero)   # true negative predictions
    false_neg = neg_pred - true_neg
    
    cm = np.array([[true_neg, false_pos], [false_neg, true_pos]])
    accuracy1 = (true_pos + true_neg) / tot_pred        
    
    pos_accuracy = true_pos / pos_pred 
    neg_accuracy = true_neg / neg_pred 
    
    # 3. F1 score
    precision = true_pos / pos_pred
    recall = true_pos / (false_neg + true_pos)
    F1 = 2* (1 / ((1/precision)+(1/recall)))
    
    # 4. Binary cross-entropy
    BCE = 0
    for s in range(0,pred_prob.shape[1]):
        for i in range(0,pred_prob.shape[0]):
            if np.isnan(pred_prob[i,s]) == False:
                if pred_prob[i,s] == 0:
                    BCE += (binary_returns[i,s]*0 + (1-binary_returns[i,s])*log(1-pred_prob[i,s]))
                elif pred_prob[i,s] == 1:
                    BCE += (binary_returns[i,s]*0 + (1-binary_returns[i,s])*0)
                else:
                    BCE += (binary_returns[i,s]*log(pred_prob[i,s]) + (1-binary_returns[i,s])*log(1-pred_prob[i,s]))
    N = sum(sum(~np.isnan(pred_prob)))
    BCE = BCE / -N
    
    # Appending results to summary table
    summary = [up_pred_share, accuracy, pos_accuracy, neg_accuracy, recall, F1, BCE]
    table.append(summary)

    
# Making a dataframe and adding headers and descrpi
table = pd.DataFrame(table)
table.columns = ['Positive predictions share', 'Directional accuracy', 'Positive accuracy', 'Negative accuracy', 'Recall', 'F1 Score', 'Binary cross-entropy']
table.index = ['Logistic', 'RAF', 'SVM', 'LSTM_i', 'LSTM_a', 'LSTM_d', 'LSTM_f']
table = round(table, 4)
table = np.transpose(table)

#table.to_csv('accuray table.csv', index=True, header=True, sep=',')



