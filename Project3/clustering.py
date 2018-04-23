# exercise 2.1.1
from tqdm import tqdm
import numpy as np
from sklearn import model_selection
from sklearn.mixture import GaussianMixture

from Tools.toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

# Global variables
classifiers = ["Decision Tree", "K-Nearest Neighbors", "Naive Bayes", "Artificial Neural Network",
               "Multinomial Logistic Regression"]

# Encode card sort and hand class in dictionary
cardSorts = {"Hearts": 1, "Spades": 2, "Diamonds": 3, "Clubs": 4}
classNames = ["Nothing", "One pair", "Two pairs", "Three of a kind", "Straight", "Flush", "Full house",
              "Four of a kind", "Straight flush", "Royal flush"]
classDict = dict(zip(range(len(classNames)), classNames))


def text_summary(class_report, conf_matrices, variable, class_type, variable_name, folds, error, error_weighted,
                 prob_dict):
    '''
    Makes a text file with classification results

    :param class_report: LIST of STRINGs containing classification reports - size=[len(folds)xlen(variable)]
    :param conf_matrices: LIST of STRINGs containing confusion matrices - size=[len(folds)xlen(variable)]
    :param variable: LIST of the values that is varied during the cross-validation process
    :param class_type: STRING containing the type of classification
    :param variable_name: STRING containing the name of the variable that is varied in cross-validation
    :param folds: INT - the number of cross-validation folds
    :return:
    '''
    with open("{0}.txt".format(class_type), 'w') as of:
        for j in range(folds):
            of.write('---------- FOLD # {0} ---------\n'.format(j + 1))
            for i, v in enumerate(variable):
                of.write('--------------------  {0} = {1}  --------------------\n'.format(variable_name, v))
                of.write(class_report[j][i])
                of.write("\n")

                of.write('--------- CONFUSION MATRIX ------------\n')
                of.write(conf_matrices[j][i])
                of.write("\n\n")

        of.write('\n--------------------------------------------------------\n\n')
        of.write('------- MISCLASSIFICATION TABLE -------\n\n')
        of.write('{0}: {1}\n'.format(variable_name, str(variable)))
        of.write('CV folds: {0}\n'.format(folds))
        of.write('\n\nMISCLASSIFICATION\n{0}'.format(str(error)))
        of.write('\n\nMISCLASSIFICATION WEIGHTED\n{0}'.format(str(error)))
        of.write('\n\nWEIGHTS\n{0}'.format(str(prob_dict)))


def classify():
    '''
    Performs a classification task on a poker hand dataset with various classification methods.
    Saves figures with errors and text files with classification summary
    :return:
    '''

    ###### LOAD ALL DATA INTO MEMORY #########################################
    print("LOADING DATA")
    df = pd.read_excel('../PokerHand_all.xlsx')
    # one out of K encoding for all categorical variables
    df = pd.get_dummies(df, columns=['Suit 1', 'Suit 2', 'Suit 3', 'Suit 4', 'Suit 5'])
    df = pd.get_dummies(df, columns=['Rank 1', 'Rank 2', 'Rank 3', 'Rank 4', 'Rank 5'])

    attributeNames = df.loc[:, df.columns != 'Hand'].columns

    # Extract class vector and variable matrix
    y = df['Hand'].values
    X = df.loc[:, df.columns != 'Hand'].values

    # Priors of each class in the entire dataset
    prob_dict = {}
    for hand in classDict.keys():
        prob_dict[hand] = len(df[(df['Hand'] == hand)]) / float(len(df))

    X = X[0:1000]
    y = y[0:1000]

    print("DATA LOADED")

    # K-fold crossvalidation
    K = 5
    CV = model_selection.KFold(n_splits=K, shuffle=True)

    k = 0

    ####################################################################################################################
    #####################################          GAUSSIAN MIXTURE MODEL          #####################################
    ####################################################################################################################
    if 0:
        # Range of K's to try
        KRange = range(1, 21)
        T = len(KRange)

        covar_type = 'diag'  # you can try out 'diag' as well
        reps = 10  # number of fits with different initalizations, best result will be kept

        # Allocate variables
        BIC = np.zeros((T,))
        AIC = np.zeros((T,))
        CVE = np.zeros((T,))

        # K-fold crossvalidation
        CV = model_selection.KFold(n_splits=10, shuffle=True)

        for t, K in enumerate(KRange):
            print('Fitting model for K={0}'.format(K))

            # Fit Gaussian mixture model
            gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X)

            # Get BIC and AIC
            BIC[t,] = gmm.bic(X)
            AIC[t,] = gmm.aic(X)

            # For each crossvalidation fold
            for train_index, test_index in CV.split(X):
                # extract training and test set for current CV fold
                X_train = X[train_index]
                X_test = X[test_index]

                # Fit Gaussian mixture model to X_train
                gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

                # compute negative log likelihood of X_test
                CVE[t] += -gmm.score_samples(X_test).sum()

        # Store results in excel
        df_score = pd.DataFrame.from_dict(data={'BIC': BIC, 'AIC': AIC, 'CVE': CVE})

        writer = pd.ExcelWriter("clustering_errors.xlsx", engine="xlsxwriter")
        df_score.to_excel(writer, sheet_name="GMM")

        # Plot results

        plt.figure(1)
        plt.plot(KRange, BIC, '-*b')
        plt.plot(KRange, AIC, '-xr')
        plt.plot(KRange, 2 * CVE, '-ok')
        plt.legend(['BIC', 'AIC', 'Crossvalidation'])
        plt.xlabel('Number of components (K)')
        plt.savefig("GMM_likelihoods.png")


    ####################################################################################################################
    #####################################          HIERARCHICAL CLUSTERING          ####################################
    ####################################################################################################################
    if 1:
        # Perform hierarchical/agglomerative clustering on data matrix
        # Method = 'single'
        # Method = 'complete'
        # Method = 'average'
        Method = 'ward'
        Metric = 'euclidean'

        Z = linkage(X, method=Method, metric=Metric)

        # Compute and display clusters by thresholding the dendrogram
        Maxclust = 4
        cls = fcluster(Z, criterion='maxclust', t=Maxclust)
        figure(1)
        clusterplot(X, cls.reshape(cls.shape[0], 1), y=y)

        # Display dendrogram
        max_display_levels = 6
        figure(2, figsize=(10, 4))
        dendrogram(Z, truncate_mode='level', p=max_display_levels)

        show()

    plt.show()


def main():
    print(dt.datetime.now())
    start = dt.datetime.now()
    classify()
    stop = dt.datetime.now()

    print("##################################################")
    print("##################################################")
    print("TIME ELAPSED: {0}".format((stop - start)))
    print("##################################################")
    print("##################################################")


if __name__ == '__main__':
    main()