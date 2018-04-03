from matplotlib.pyplot import figure, plot, subplot, title, show, bar
import numpy as np
from scipy.io import loadmat
import neurolab as nl
from sklearn import model_selection
from scipy import stats
import pandas as pd
import xlrd

# Load xls sheet with data
doc = xlrd.open_workbook('../PokerHand_all.xlsx').sheet_by_index(0)

# Encode card sort and hand class in dictionary
cardSorts = {"Hearts": 1, "Spades": 2, "Diamonds": 3, "Clubs": 4}
classNames = ["Nothing", "One pair", "Two pairs", "Three of a kind", "Straight", "Flush", "Full house",
              "Four of a kind", "Straight flush", "Royal flush"]
classDict = dict(zip(classNames, range(5)))

# Compute v*alues of N, M and C.
N = doc.nrows - 1  # number of rows in the dataset
M = doc.ncols - 1  # number of columns in dataset
C = len(classNames)

#Get names of  every attribute
attributeNames = doc.row_values(0, 0, M)

# Extract vector y, convert to NumPy matrix and transpose
y = np.mat(doc.col_values(doc.ncols - 1, 1, N)).T

# Preallocate memory, then extract excel data to matrix X
X = np.mat(np.empty((N - 1, M)))
for i, col_id in enumerate(range(0, M)):
    X[:, i] = np.mat(doc.col_values(col_id, 1, N)).T
                
#    # Temporary data -------------------------
X = X[0:10000]    
y = y[0:10000]
N = 10000
    #-----------------------------------------
## Normalize and compute PCA (UNCOMMENT to experiment with PCA preprocessing)
#Y = stats.zscore(X,0);
#U,S,V = np.linalg.svd(Y,full_matrices=False)
#V = V.T
##Components to be included as features
#k_pca = 3
#X = X @ V[:,0:k_pca]
#N, M = X.shape


# Parameters for neural network classifier
n_hidden_units = 10      # number of hidden units
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
    break
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


#% The weights if the network can be extracted via
#bestnet[0].layers[0].np['w'] # Get the weights of the first layer
#bestnet[0].layers[0].np['b'] # Get the bias of the first layer



