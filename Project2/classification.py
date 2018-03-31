# exercise 2.1.1
from tqdm import tqdm
import pandas as pd
import numpy as np
import xlrd
from sklearn import model_selection, tree, naive_bayes, neighbors, neural_network, linear_model

import graphviz
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt
from scipy.stats.mstats import zscore

#Global variables
classifiers = ["Decision Tree", "K-Nearest Neighbors", "Naive Bayes", "Artificial Neural Network", "Multinomial/Logistic Regression"]

# Encode card sort and hand class in dictionary
cardSorts = {"Hearts": 1, "Spades": 2, "Diamonds": 3, "Clubs": 4}
classNames = ["Nothing", "One pair", "Two pairs", "Three of a kind", "Straight", "Flush", "Full house",
              "Four of a kind", "Straight flush", "Royal flush"]
classDict = dict(zip(range(len(classNames)), classNames))



def mat2latex(X):
    '''
    Converts a matrix/array into the latex representation of it at prints it to console.
    :param X: a matrix/array of numbers
    :return:
    '''

    rows = np.ma.size(X, 0)
    cols = np.ma.size(X, 1)
    # matrix format
    curr = "\[ \n"
    curr = "{0} X = ".format(curr)
    curr = "%s \left[ \\begin{array} {*{ %d }c} \n"%(curr, cols)
    # row data

    for ii in range(rows):
        rowstr = ""
        for jj in range(cols-1):
            rowstr += "{:.2f} & ".format(X[ii,jj])
        rowstr += "{:.2f} \\\\ \n".format(X[ii,cols-1])
        curr += rowstr

    curr = "%s \end{array}\\right]\n \]"%curr
    print("#######################################################################")
    print("##################### MATRIX TO LATEX OUTPUT ##########################")
    print()
    print(curr)
    print()
    print("#######################################################################")
    print()



def main():
    # df = pd.read_excel('../PokerHand_all.xlsx')
    # print(len(df))

    ###### LOAD ALL DATA INTO MEMORY #########################################
    print("LOADING DATA")
    # # Load xls sheet with data
    # doc = xlrd.open_workbook('../PokerHand_all.xlsx').sheet_by_index(0)
    #
    #
    # # Compute v*alues of N, M and C.
    # N = doc.nrows - 1  # number of rows in the dataset
    # M = doc.ncols - 1  # number of columns in dataset
    # C = len(classNames)
    #
    # #Get names of  every attribute
    # attributeNames = doc.row_values(0, 0, M)
    #
    # # Extract vector y, convert to NumPy matrix and transpose
    # y = np.mat(doc.col_values(doc.ncols - 1, 1, N)).T
    #
    # df_all = pd.DataFrame.from_dict(data={"values": pd.Series(np.asarray(y).squeeze())})
    #
    # prob_dict = {}
    # for hand in classDict.keys():
    #     prob_dict[hand] = len(df_all[(df_all['values']==hand)])/float(len(df_all))
    #
    # # Preallocate memory, then extract excel data to matrix X
    # X = np.mat(np.empty((N - 1, M)))
    # for i, col_id in enumerate(range(0, M)):
    #     X[:, i] = np.mat(doc.col_values(col_id, 1, N)).T
    #
    # # Standardize data by removing the mean
    # X_hat =  X - np.ones((N-1,1))*X.mean(0) #zscore(X)
    #####################################################################
    # X = X[0:1000, :]
    # y = y[0:1000    ]

    df = pd.read_excel('../PokerHand_all.xlsx')
    df = pd.get_dummies(df, columns=['Suit 1', 'Suit 2', 'Suit 3', 'Suit 4', 'Suit 5'])
    df = pd.get_dummies(df, columns=['Rank 1', 'Rank 2', 'Rank 3', 'Rank 4', 'Rank 5'])
    # print(df.head())
    y = df['Hand'].values
    X = df.loc[:, df.columns != 'Hand'].values

    prob_dict = {}
    for hand in classDict.keys():
        prob_dict[hand] = len(df[(df['Hand'] == hand)])/float(len(df))

    print("DATA LOADED")

    X = X[0:1000]
    # print(X)
    y = y[0:1000]

    # K-fold crossvalidation
    K = 5
    CV = model_selection.KFold(n_splits=K, shuffle=True)

    # # Initialize variable
    # Error_train = np.empty((len(tc), K))
    # Error_test = np.empty((len(tc), K))

    k = 0

    # # Simple holdout-set crossvalidation
    # test_proportion = 0.9
    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_pr oportion)




    ################################################################################################################
    #####################################            DECISION TREE             #####################################
    ################################################################################################################
    if 0:
        print("Decision Tree Classification")

        # ADJUST TREE PARAMETERS (DEPTH AND MIN_SAMPLES)
        # Tree complexity parameter - constraint on maximum depth
        tc = np.arange(5, 51, 5)
        tc = tc.tolist()
        tc.extend(np.arange(75, 500, 25))

        # Initialize variables
        Error_train_weighted = np.empty((len(classifiers), len(tc), K))
        Error_test_weighted = np.empty((len(classifiers), len(tc), K))
        Error_train = np.empty((len(classifiers), len(tc), K))
        Error_test = np.empty((len(classifiers), len(tc), K))

        for train_index, test_index in CV.split(X, y):
            print('Computing CV fold: {0}/{1}..'.format(k + 1, K))

            # extract training and test set for current CV fold
            X_train, y_train = X[train_index, :], y[train_index]
            X_test, y_test = X[test_index, :], y[test_index]

            # for i, t in enumerate(tc):
            for i, t in enumerate(tqdm(tc)):
                # Fit decision tree classifier, Gini split criterion, different pruning levels
                dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=t)# max_depth=t)
                dtc = dtc.fit(X_train, y_train)

                y_est_test = dtc.predict(X_test)
                y_est_train = dtc.predict(X_train)

                df_class_train = pd.DataFrame.from_dict(
                    data={"value": pd.Series(np.asarray(y_train).squeeze()), "estimate": pd.Series(y_est_train)})
                df_class_test = pd.DataFrame.from_dict(
                    data={"value": pd.Series(np.asarray(y_test).squeeze()), "estimate": pd.Series(y_est_test)})

                Error_test[0, i, k] = len(df_class_test[(df_class_test['estimate'] != df_class_test['value'])]) / float(len(df_class_test))
                Error_train[0, i, k] = len(df_class_train[(df_class_train['estimate'] != df_class_train['value'])]) / float(len(df_class_train))

                Error_test_weighted[0, i, k] = score_func(df_class_test, prob_dict)
                Error_train_weighted[0, i, k] = score_func(df_class_train, prob_dict)

            k += 1

        f = plt.figure()
        plt.title(classifiers[0])
        plt.boxplot(Error_test_weighted[0].T)
        plt.xlabel('Model complexity (max tree depth)')
        plt.ylabel('Test error across CV folds, K={0})'.format(K))

        f = plt.figure()
        plt.title(classifiers[0])
        plt.plot(tc, Error_train_weighted[0].mean(1), 'b-')
        plt.plot(tc, Error_test_weighted[0].mean(1), 'r-')
        plt.plot(tc, Error_train[0].mean(1), 'b--')
        plt.plot(tc, Error_test[0].mean(1), 'r-.')
        plt.xlabel('Model complexity (max tree depth)')
        plt.ylabel('Error (misclassification rate, CV K={0})'.format(K))
        plt.legend(['Weighted Training Error', 'Weighted Testing Error', 'Un-Weighted Training Error',
                    'Un-Weighted Testing Error'])

    ################################################################################################################
    ################################################################################################################

    ################################################################################################################
    ##################################            K-Nearest Neighbors             ##################################
    ################################################################################################################
    if 0:
        print("K-Nearest Neighbor Classification")
        k=0
        # ADJUST NUMBER OF NEIGHBORS
        nn = np.arange(5, 51, 5)

        # Initialize variables
        Error_train_weighted = np.empty((len(classifiers), len(nn), K))
        Error_test_weighted = np.empty((len(classifiers), len(nn), K))
        Error_train = np.empty((len(classifiers), len(nn), K))
        Error_test = np.empty((len(classifiers), len(nn), K))

        for train_index, test_index in CV.split(X, y):
            print('\nComputing CV fold: {0}/{1}..'.format(k + 1, K))

            # extract training and test set for current CV fold
            X_train, y_train = X[train_index, :], y[train_index]
            X_test, y_test = X[test_index, :], y[test_index]

            # for i, n in enumerate(nn):
            for i,n in enumerate(tqdm(nn)):
                # Fit K-Nearest Neighbor classifier
                # print("TRAINING MODEL")
                knbc = neighbors.KNeighborsClassifier(n_neighbors=n)
                knbc = knbc.fit(X_train, np.asarray(y_train).squeeze())

                # print("PREDICTING TESTING SET")
                y_est_test = knbc.predict(X_test)
                # print("PREDICTING TRAINING SET")
                y_est_train = knbc.predict(X_train)
                # y_est_train = np.asarray(y_train).squeeze()

                df_class_train = pd.DataFrame.from_dict(data={"value": pd.Series(np.asarray(y_train).squeeze()), "estimate": pd.Series(y_est_train)})
                df_class_test = pd.DataFrame.from_dict(data={"value": pd.Series(np.asarray(y_test).squeeze()), "estimate": pd.Series(y_est_test)})

                Error_test_weighted[1, i, k] = score_func(df_class_test, prob_dict)
                Error_train_weighted[1, i, k] = score_func(df_class_train, prob_dict)

                Error_test[1, i, k] = len(df_class_test[(df_class_test['estimate'] != df_class_test['value'])]) / float(len(df_class_test))
                Error_train[1, i, k] = len(df_class_train[(df_class_train['estimate'] != df_class_train['value'])]) / float(len(df_class_train))

            k += 1

        f = plt.figure()
        plt.title(classifiers[1])
        plt.boxplot(Error_test_weighted[1].T)
        plt.xlabel('Model complexity (# of neighbors)')
        plt.ylabel('Test error across CV folds, K={0})'.format(K))

        f = plt.figure()
        plt.title(classifiers[1])
        plt.plot(nn, Error_train_weighted[1].mean(1), 'b-')
        plt.plot(nn, Error_test_weighted[1].mean(1), 'r-')
        plt.plot(nn, Error_train[1].mean(1), 'b--')
        plt.plot(nn, Error_test[1].mean(1), 'r-.')
        plt.xlabel('Model complexity (# of neighbors)')
        plt.ylabel('Error (misclassification rate, CV K={0})'.format(K))
        plt.legend(['Weighted Training Error', 'Weighted Testing Error', 'Un-Weighted Training Error', 'Un-Weighted Testing Error'])

    ################################################################################################################
    ################################################################################################################

    ################################################################################################################
    ######################################            Naive Bayes             ######################################
    ################################################################################################################
    if 0:
        print("Naive Bayes Classification")
        k = 0
        # ADJUST NUMBER OF NEIGHBORS
        alpha = np.arange(0.0, 1.1, 0.1)

        # Initialize variables
        Error_train_weighted = np.empty((len(classifiers), len(alpha), K))
        Error_test_weighted = np.empty((len(classifiers), len(alpha), K))
        Error_train = np.empty((len(classifiers), len(alpha), K))
        Error_test = np.empty((len(classifiers), len(alpha), K))

        for train_index, test_index in CV.split(X, y):
            print('\nComputing CV fold: {0}/{1}..'.format(k + 1, K))

            # extract training and test set for current CV fold
            X_train, y_train = X[train_index, :], y[train_index]
            X_test, y_test = X[test_index, :], y[test_index]

            df_priors = pd.DataFrame.from_dict(data={"values": pd.Series(np.asarray(y_train).squeeze())})
            priors = []
            # print(classDict.keys())
            # print(np.sort(classDict.keys(), axis=1))
            for classNr in np.sort(np.unique(df_priors['values'])):
                priors.append(len(df_priors[(df_priors['values'] == classNr)])/float(len(df_priors)))


            # for i, n in enumerate(nn):
            for i, a in enumerate(tqdm(alpha)):
                # Fit K-Nearest Neighbor classifier
                # print("TRAINING MODEL")
                nbc = naive_bayes.MultinomialNB(alpha=a, fit_prior=False, class_prior=priors)
                nbc = nbc.fit(X_train, np.asarray(y_train).squeeze())

                # print("PREDICTING TESTING SET")
                y_est_test = nbc.predict(X_test)
                # print("PREDICTING TRAINING SET")
                y_est_train = nbc.predict(X_train)
                # y_est_train = np.asarray(y_train).squeeze()

                df_class_train = pd.DataFrame.from_dict(
                    data={"value": pd.Series(np.asarray(y_train).squeeze()), "estimate": pd.Series(y_est_train)})
                df_class_test = pd.DataFrame.from_dict(
                    data={"value": pd.Series(np.asarray(y_test).squeeze()), "estimate": pd.Series(y_est_test)})

                Error_test_weighted[2, i, k] = score_func(df_class_test, prob_dict)
                Error_train_weighted[2, i, k] = score_func(df_class_train, prob_dict)

                Error_test[2, i, k] = len(
                    df_class_test[(df_class_test['estimate'] != df_class_test['value'])]) / float(
                    len(df_class_test))
                Error_train[2, i, k] = len(
                    df_class_train[(df_class_train['estimate'] != df_class_train['value'])]) / float(
                    len(df_class_train))

            k += 1

        f = plt.figure()
        plt.title(classifiers[2])
        plt.boxplot(Error_test_weighted[2].T)
        plt.xlabel('Model complexity (Smoothing constant)')
        plt.ylabel('Test error across CV folds, K={0})'.format(K))

        f = plt.figure()
        plt.title(classifiers[2])
        plt.plot(alpha, Error_train_weighted[2].mean(1), 'b-')
        plt.plot(alpha, Error_test_weighted[2].mean(1), 'r-')
        plt.plot(alpha, Error_train[2].mean(1), 'b--')
        plt.plot(alpha, Error_test[2].mean(1), 'r-.')
        plt.xlabel('Model complexity (Smoothing constant)')
        plt.ylabel('Error (misclassification rate, CV K={0})'.format(K))
        plt.legend(['Weighted Training Error', 'Weighted Testing Error', 'Un-Weighted Training Error',
                    'Un-Weighted Testing Error'])

    ################################################################################################################
    ################################################################################################################

    ################################################################################################################
    ###############################            Artificial Neural Network             ###############################
    ################################################################################################################
    if 0:
        print("Artificial Neural Network Classification")

        k = 0
        # ADJUST NUMBER OF NEIGHBORS
        NHiddenUnits = np.arange(10, 101, 10);  # <-- Try to change this, what happens? why?
        hidden = []
        for q in NHiddenUnits:
            for e in NHiddenUnits:
                hidden.append((q, e))

        # Initialize variables
        Error_train_weighted = np.empty((len(classifiers), len(hidden), K))
        Error_test_weighted = np.empty((len(classifiers), len(hidden), K))
        Error_train = np.empty((len(classifiers), len(hidden), K))
        Error_test = np.empty((len(classifiers), len(hidden), K))

        for train_index, test_index in CV.split(X, y):
            print('\nComputing CV fold: {0}/{1}..'.format(k + 1, K))

            # extract training and test set for current CV fold
            X_train, y_train = X[train_index, :], y[train_index]
            X_test, y_test = X[test_index, :], y[test_index]

            # for i, n in enumerate(nn):
            for i, n in enumerate(tqdm(hidden)):
                # %% Model fitting and prediction
                ## ANN Classifier, i.e. MLP with one hidden layer
                clf = neural_network.MLPClassifier(solver='adam', alpha=1e-4, activation="tanh",
                                                   hidden_layer_sizes=n, random_state=1)
                clf.fit(X_train, np.asarray(y_train).squeeze())

                # print("PREDICTING TESTING SET")
                y_est_test = clf.predict(X_test)
                # print("PREDICTING TRAINING SET")
                y_est_train = clf.predict(X_train)
                # y_est_train = np.asarray(y_train).squeeze()

                df_class_train = pd.DataFrame.from_dict(
                    data={"value": pd.Series(np.asarray(y_train).squeeze()), "estimate": pd.Series(y_est_train)})
                df_class_test = pd.DataFrame.from_dict(
                    data={"value": pd.Series(np.asarray(y_test).squeeze()), "estimate": pd.Series(y_est_test)})

                Error_test_weighted[3, i, k] = score_func(df_class_test, prob_dict)
                Error_train_weighted[3, i, k] = score_func(df_class_train, prob_dict)
                # print(clf.predict(X_test) != y_test)
                Error_test[3, i, k] = len(
                    df_class_test[(df_class_test['estimate'] != df_class_test['value'])]) / float(
                    len(df_class_test))
                Error_train[3, i, k] = len(
                    df_class_train[(df_class_train['estimate'] != df_class_train['value'])]) / float(
                    len(df_class_train))

                # print('Number of miss-classifications for ANN:\n\t {0} out of {1}'.format(
                #     len(df_class_test[(df_class_test['estimate'] != df_class_test['value'])]),
                #     len(df_class_test)))

            k += 1

        f = plt.figure()
        plt.title(classifiers[3])
        plt.boxplot(Error_test_weighted[3].T)
        plt.xlabel('Model complexity (# of units in hidden layers)')
        plt.ylabel('Test error across CV folds, K={0})'.format(K))

        f = plt.figure()
        plt.title(classifiers[3])
        plt.plot(range(len(hidden)), Error_train_weighted[3].mean(1), 'b-')
        plt.plot(range(len(hidden)), Error_test_weighted[3].mean(1), 'r-')
        plt.plot(range(len(hidden)), Error_train[3].mean(1), 'b--')
        plt.plot(range(len(hidden)), Error_test[3].mean(1), 'r-.')
        plt.xticks(range(len(hidden)), [str(x) for x in hidden], rotation=90)
        plt.xlabel('Model complexity (# of units in hidden layers)')
        plt.ylabel('Error (misclassification rate, CV K={0})'.format(K))
        plt.legend(['Weighted Training Error', 'Weighted Testing Error', 'Un-Weighted Training Error',
                    'Un-Weighted Testing Error'])

    ################################################################################################################
    ################################################################################################################

    plt.show()



def score_func(dataframe, weights):
    df_error = dataframe[(dataframe['value'] != dataframe["estimate"])]
    error_rate = 0.0
    for poker_hand in classDict.keys():
        error_rate += len(df_error[(df_error['value'] == poker_hand)])/len(dataframe) * weights[poker_hand]#len(dataframe[(dataframe['value'] == poker_hand)]) / float(len(dataframe))

    return error_rate


if __name__ == '__main__':
    main()