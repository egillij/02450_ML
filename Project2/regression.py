# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xlrd
from scipy.linalg import svd
from matplotlib.pyplot import figure, plot, subplot, suptitle, title, xlabel, ylabel, show, clim, xticks
from scipy.stats.mstats import zscore
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot



# Encode card sort and hand class in dictionary
cardSorts = {"Hearts": 1, "Spades": 2, "Diamonds": 3, "Clubs": 4}
classNames = ["Nothing", "One pair", "Two pairs", "Three of a kind", "Straight", "Flush", "Full house",
              "Four of a kind", "Straight flush", "Royal flush"]
classDict = dict(zip(range(len(classNames)), classNames))

def main():

    ###### LOAD ALL DATA INTO MEMORY #########################################
#    print("LOADING DATA")
#     # Load xls sheet with data
#    doc = xlrd.open_workbook('../PokerHand_all.xlsx').sheet_by_index(0)
#    
#    
#    # Compute v*alues of N, M and C.
#    N = doc.nrows - 1  # number of rows in the dataset
#    M = doc.ncols - 1  # number of columns in dataset
#    C = len(classNames)
#    
#     #Get names of  every attribute
#    attributeNames = doc.row_values(0, 0, M)
#    
#     # Extract vector y, convert to NumPy matrix and transpose
#    y = np.mat(doc.col_values(doc.ncols - 1, 1, N)).T
#    
#    df_all = pd.DataFrame.from_dict(data={"values": pd.Series(np.asarray(y).squeeze())})
#    
#    prob_dict = {}
#    for hand in classDict.keys():
#        prob_dict[hand] = len(df_all[(df_all['values']==hand)])/float(len(df_all))
#    
#     # Preallocate memory, then extract excel data to matrix X
#    X = np.mat(np.empty((N - 1, M)))
#    for i, col_id in enumerate(range(0, M)):
#        X[:, i] = np.mat(doc.col_values(col_id, 1, N)).T
#    
#     # Standardize data by removing the mean
#    X_hat =  X - np.ones((N-1,1))*X.mean(0) #zscore(X)
    #####################################################################
    
    df = pd.read_excel('../PokerHand_all.xlsx')
    df = pd.get_dummies(df, columns=['Suit 1', 'Suit 2', 'Suit 3', 'Suit 4', 'Suit 5'])
    df = pd.get_dummies(df, columns=['Rank 1', 'Rank 2', 'Rank 3', 'Rank 4', 'Rank 5'])
    # print(df.head())
    y = df['Hand'].values
    X = df.loc[:, df.columns != 'Hand'].values
    
    
    # Compute values of N, M and C.
    N, M = X.shape # number of rows,columns in the dataset
    C = len(classNames)
    
    prob_dict = {}
    for hand in classDict.keys():
        prob_dict[hand] = len(df[(df['Hand'] == hand)])/float(len(df))
    
    #Get names of  every attribute
    attributeNames = list(df)
#    
#    
#    
#    # Temporary data -------------------------
#    X = X[0:10000]    
#    y = y[0:10000]
#    
#    N = 10000
    #-----------------------------------------
    
    ## Crossvalidation
    # Create crossvalidation partition for evaluation
    K = 5
    CV = model_selection.KFold(n_splits=K,shuffle=True)    
    
    
    newX = np.concatenate((np.ones((X.shape[0],1)),X),1)
    newX[:,2] = np.power(newX[:,2],2)  
    
    newAttributeNames = [u'Constant']+attributeNames
    newM = len(attributeNames)+1
    
    
    
    k = 0
    
    # Initialize variables
    Features = np.zeros((newM,K))
    Error_train = np.empty((K,1))
    Error_test = np.empty((K,1))
    Error_train_fs = np.empty((K,1))
    Error_test_fs = np.empty((K,1))
    Error_train_nofeatures = np.empty((K,1))
    Error_test_nofeatures = np.empty((K,1))
    
    
    best_error = 1e32
    best_k = 0
    best_loss = []
    best_rec = []
    best_selected_features = []
    best_params = []
    
    
    for train_index, test_index in CV.split(X):
        
        # extract training and test set for current CV fold
        X_train = newX[train_index,:]
        y_train = y[train_index]
        X_test = newX[test_index,:]
        y_test = y[test_index]
        internal_cross_validation = 5
        
        # Compute squared error without using the input data at all
        Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
        Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]
        
        # Compute squared error with all features selected (no feature selection)
        m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
        Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
        Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
    
        # Compute squared error with feature subset selection
        selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation)
    
        Features[selected_features,k]=1
    
        if len(selected_features) is 0:
            print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
        else:
            m = lm.LinearRegression(fit_intercept=True).fit(X_train[:,selected_features], y_train)
            Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
            Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]
        
            figure(k)
            subplot(1,2,1)
            plot(range(1,len(loss_record)), loss_record[1:])
            xlabel('Iteration')
            ylabel('Squared error (crossvalidation)')    
            
            subplot(1,3,3)
            bmplot(newAttributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
            clim(-1.5,0)
            xlabel('Iteration')
            
            if Error_test_fs[k] < best_error:
                best_error = Error_test_fs[k]
                best_k = k
                best_loss = loss_record
                best_rec = features_record
                best_selected_features = selected_features
                best_params = m.coef_
    
        print('Cross validation fold {0}/{1}'.format(k+1,K))
        print('Train indices: {0}'.format(train_index))
        print('Test indices: {0}'.format(test_index))
        print('Features no: {0}\n'.format(selected_features.size))
    
        k+=1
    
    
    # Display results
    print('\n')
    print('Linear regression without feature selection:\n')
    print('- Training error: {0}'.format(Error_train.mean()))
    print('- Test error:     {0}'.format(Error_test.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
    print('Linear regression with feature selection:\n')
    print('- Training error: {0}'.format(Error_train_fs.mean()))
    print('- Test error:     {0}'.format(Error_test_fs.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))
    
    figure(k)
    subplot(1,3,2)
    bmplot(newAttributeNames, range(1,Features.shape[1]+1), -Features)
    clim(-1.5,0)
    xlabel('Crossvalidation fold')
    ylabel('Attribute')
    
    
    # Inspect selected feature coefficients effect on the entire dataset and
    # plot the fitted model residual error as function of each attribute to
    # inspect for systematic structure in the residual
    
    f=2 # cross-validation fold to inspect
    ff=Features[:,f-1].nonzero()[0]
    if len(ff) is 0:
        print('\nNo features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        m = lm.LinearRegression(fit_intercept=True).fit(newX[:,ff], y)
        
        y_est= m.predict(newX[:,ff])
        residual=y-y_est
        
        figure(k+1, figsize=(12,6))
        title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
        for i in range(0,len(ff)):
           subplot(2,np.ceil(len(ff)/2.0),i+1)
           plot(newX[:,ff[i]],residual,'.')
           xlabel(newAttributeNames[ff[i]])
           ylabel('residual error')
        
        
    show()


if __name__ == '__main__':
    main()
