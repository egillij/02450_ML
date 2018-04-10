# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import neurolab as nl
from scipy import stats
from scipy.linalg import svd
from matplotlib.pyplot import figure, plot, boxplot, subplot, suptitle, title, xlabel, ylabel, show, clim, xticks, bar
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
    df = pd.read_excel('../PokerHand_all.xlsx')
    
    df = pd.get_dummies(df, columns=['Suit 1', 'Suit 2', 'Suit 3', 'Suit 4', 'Suit 5'])
    df = pd.get_dummies(df, columns=['Rank 1', 'Rank 2', 'Rank 3', 'Rank 4', 'Rank 5'])
    
    y = df['Hand'].values
    X = df.loc[:, df.columns != 'Hand'].values
    
    
    # Compute values of N, M and C.
    N, M = X.shape # number of rows,columns in the dataset
    C = len(classNames)
    attributeNames = list(df.loc[:, df.columns != 'Hand'])
    
    # Small subset -------------------------
#    X = X[0:50000]    
#    y = y[0:50000]
#    
#    N = 50000
    #-----------------------------------------
    
    ## Crossvalidation
    # Create crossvalidation partition for evaluation
    K = 5
    CV = model_selection.KFold(n_splits=K,shuffle=True) 
    
    Error_linear_re = np.empty((K,1))
    Error_ann = np.empty((K,1))
    
    
    ################################################################################################################
    #####################################            Linear Regression             #################################
    ################################################################################################################
    if 1:
        # Initialize variables
        Features = np.zeros((M,K))
        Error_train = np.empty((K,1))
        Error_test = np.empty((K,1))
        Error_train_fs = np.empty((K,1))
        Error_test_fs = np.empty((K,1))
        Error_train_nofeatures = np.empty((K,1))
        Error_test_nofeatures = np.empty((K,1))
        
        k=0
        for train_index, test_index in CV.split(X):
        
            # extract training and test set for current CV fold
            X_train = X[train_index,:]
            y_train = y[train_index]
            X_test = X[test_index,:]
            y_test = y[test_index]
            internal_cross_validation = 10
            
            # Compute squared error without using the input data at all
            Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
            Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]
        
            # Compute squared error with all features selected (no feature selection)
            m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
            Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
            Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
        
            # Compute squared error with feature subset selection
            #textout = 'verbose';
            textout = '';
            selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,display=textout)
            
            Features[selected_features,k]=1
            # .. alternatively you could use module sklearn.feature_selection
            if len(selected_features) is 0:
                print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
            else:
                m = lm.LinearRegression(fit_intercept=True).fit(X_train[:,selected_features], y_train)
                Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
                Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]

                y_linear_re = m.predict(X_test[:,selected_features])
                Error_linear_re[k] = 100*(y_linear_re!=y_test).sum().astype(float)/len(y_test)
            
                figure(k)
                subplot(1,2,1)
                plot(range(1,len(loss_record)), loss_record[1:])
                xlabel('Iteration')
                ylabel('Squared error (crossvalidation)')    
                
                subplot(1,3,3)
                bmplot(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
                clim(-1.5,0)
                xlabel('Iteration')
        
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
        
        figure(k, figsize=(18,18))
        subplot(1,3,2)
        bmplot(attributeNames, range(1,Features.shape[1]+1), -Features)
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
            m = lm.LinearRegression(fit_intercept=True).fit(X[:,ff], y)
            
            print("Coefficients: "  + m.coef_)
            
            y_est= m.predict(X[:,ff])
            residual=y-y_est
            
            figure(k+1, figsize=(15,9))
            title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
            for i in range(0,len(ff)):
               subplot(2,np.ceil(len(ff)/2.0),i+1)
               plot(X[:,ff[i]],residual,'.')
               xlabel(attributeNames[ff[i]])
               ylabel('residual error')
            
            
        show()
        
    
    ################################################################################################################
    #####################################            Artificial neural network             #########################
    ################################################################################################################
    if 1:
        # Parameters for neural network classifier
        n_hidden_units = 2      # number of hidden units
        n_train = 3            # number of networks trained in each k-fold
        learning_goal = 0.5  # stop criterion 1 (train mse to be reached)
        max_epochs = 100        # stop criterion 2 (max epochs in training)
        show_error_freq = 200     # frequency of training status updates
        
        # K-fold crossvalidation
        K = 5                   # only five folds to speed up this example
        CV = model_selection.KFold(K,shuffle=True)
        
        # Variable for classification error
        errors = np.zeros(K)
        error_hist = np.zeros((max_epochs,K))
        bestnet = list()
        k=0
        for train_index, test_index in CV.split(X,y):
            print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
            
            # extract training and test set for current CV fold
            X_train = X[train_index,:]
            y_train = y[train_index]
            X_test = X[test_index,:]
            y_test = y[test_index]
            
            best_train_error = 1e100
            for i in range(n_train):
                print('Training network {0}/{1}...'.format(i+1,n_train))
                # Create randomly initialized network with 2 layers
                ann = nl.net.newff([[-3, 3]]*M, [n_hidden_units, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
                if i==0:
                    bestnet.append(ann)
                # train network
                train_error = ann.train(X_train, y_train.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
                if train_error[-1]<best_train_error:
                    bestnet[k]=ann
                    best_train_error = train_error[-1]
                    error_hist[range(len(train_error)),k] = train_error
        
            print('Best train error: {0}...'.format(best_train_error))
            y_est = bestnet[k].sim(X_test).squeeze()
            errors[k] = np.power(y_est-y_test,2).sum().astype(float)/y_test.shape[0]
            
            Error_ann[k] = 100*(y_est!=y_test).sum().astype(float)/len(y_test)
            
            k+=1
            
            
        
        # Print the average least squares error
        print('Mean-square error: {0}'.format(np.mean(errors)))
        
        figure(figsize=(6,7));
        subplot(2,1,1); bar(range(0,K),errors); title('Mean-square errors');
        subplot(2,1,2); plot(error_hist); title('Training error as function of BP iterations');
        figure(figsize=(6,7));
        subplot(2,1,1); plot(y_est); plot(y_test); title('Last CV-fold: est_y vs. test_y'); 
        subplot(2,1,2); plot((y_est-y_test)); title('Last CV-fold: prediction error (est_y-test_y)'); 
        show()    
    
    ################################################################################################################
    #####################################            Average of the training data             ######################
    ################################################################################################################
    if 1:
         # K-fold crossvalidation
        K = 5                   # only five folds to speed up this example
        CV = model_selection.KFold(K,shuffle=True)
        
        # Variable for classification error
        errors = np.zeros(K)

        k=0
        for train_index, test_index in CV.split(X,y):
            print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
            
            # extract training and test set for current CV fold
            X_train = X[train_index,:]
            y_train = y[train_index]
            X_test = X[test_index,:]
            y_test = y[test_index]
            
            # Get the average of the training data
            y_est = np.mean(y_train)
            errors[k] = np.power(y_est-y_test,2).sum().astype(float)/y_test.shape[0]
            
            k+=1
            
        print('Mean-square error: {0}'.format(np.mean(errors)))  
        
        
        
    # Test if classifiers are significantly different using methods in section 9.3.3
    # by computing credibility interval. Notice this can also be accomplished by computing the p-value using
    # [tstatistic, pvalue] = stats.ttest_ind(Error_logreg,Error_ann)
    # and test if the p-value is less than alpha=0.05. 
    z = (Error_linear_re-Error_ann)
    zb = z.mean()
    nu = K-1
    sig =  (z-zb).std()  / np.sqrt(K-1)
    alpha = 0.05
    
    zL = zb + sig * stats.t.ppf(alpha/2, nu);
    zH = zb + sig * stats.t.ppf(1-alpha/2, nu);
    
    if zL <= 0 and zH >= 0 :
        print('Classifiers are not significantly different')        
    else:
        print('Classifiers are significantly different.')
        
    # Boxplot to compare classifier error distributions
    figure()
    boxplot(np.concatenate((Error_linear_re, Error_ann),axis=1))
    xlabel('Linear Regression   vs.   Artificial Neural network')
    ylabel('Cross-validation error [%]')
    
    show()



if __name__ == '__main__':
    main()
