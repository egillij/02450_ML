# exercise 2.1.1
from tqdm import tqdm
import numpy as np
from sklearn import model_selection, tree, naive_bayes, neighbors, neural_network, linear_model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import graphviz
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

#Global variables
classifiers = ["Decision Tree", "K-Nearest Neighbors", "Naive Bayes", "Artificial Neural Network", "Multinomial Logistic Regression"]

# Encode card sort and hand class in dictionary
cardSorts = {"Hearts": 1, "Spades": 2, "Diamonds": 3, "Clubs": 4}
classNames = ["Nothing", "One pair", "Two pairs", "Three of a kind", "Straight", "Flush", "Full house",
              "Four of a kind", "Straight flush", "Royal flush"]
classDict = dict(zip(range(len(classNames)), classNames))


def text_summary(class_report, conf_matrices, variable, class_type, variable_name, folds, error, error_weighted, prob_dict):
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
            of.write('---------- FOLD # {0} ---------\n'.format(j+1))
            for i, v in     enumerate(variable):
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
    #one out of K encoding for all categorical variables
    df = pd.get_dummies(df, columns=['Suit 1', 'Suit 2', 'Suit 3', 'Suit 4', 'Suit 5'])
    df = pd.get_dummies(df, columns=['Rank 1', 'Rank 2', 'Rank 3', 'Rank 4', 'Rank 5'])

    attributeNames = df.loc[:, df.columns != 'Hand'].columns

    #Extract class vector and variable matrix
    y = df['Hand'].values
    X = df.loc[:, df.columns != 'Hand'].values

    #Priors of each class in the entire dataset
    prob_dict = {}
    for hand in classDict.keys():
        prob_dict[hand] = len(df[(df['Hand'] == hand)])/float(len(df))

    # X = X[0:1000]
    # y = y[0:1000]

    print("DATA LOADED")

    # K-fold crossvalidation
    K = 5
    CV = model_selection.KFold(n_splits=K, shuffle=True)

    k = 0

    ################################################################################################################
    #####################################            DECISION TREE             #####################################
    ################################################################################################################
    if 1:
        print("Decision Tree Classification")
        print(dt.datetime.now())

        # ADJUST TREE PARAMETERS (MIN SAMPLES FOR SPLIT)
        # tc = np.arange(5, 25, 2)
        # tc = tc.tolist()
        # tc.extend(np.arange(25, 150, 10))

        #ADJUST TREE DEPTH
        tc = np.arange(5, 200, 5)

        # Initialize variables
        Error_train_weighted = np.empty((len(tc), K))
        Error_test_weighted = np.empty((len(tc), K))
        Error_train = np.empty((len(tc), K))
        Error_test = np.empty((len(tc), K)),

        class_report_test = []
        class_report_train = []

        conf_matrices_test = []
        conf_matrices_train = []

        for train_index, test_index in CV.split(X, y):
            print('Computing CV fold: {0}/{1}..'.format(k + 1, K))

            # extract training and test set for current CV fold
            X_train, y_train = X[train_index, :], y[train_index]
            X_test, y_test = X[test_index, :], y[test_index]

            cl_report_tst = []
            cl_report_tr = []

            conf_mat_tst = []
            conf_mat_tr = []

            # Train and test all tree d
            for i, t in enumerate(tqdm(tc)):
                # Fit decision tree classifier
                dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)#min_samples_split=t)# max_depth=t
                dtc = dtc.fit(X_train, y_train)

                # try:
                #     #Visualize decision tree
                #     graph = graphviz.Source(tree.export_graphviz(dtc, out_file=None, feature_names=attributeNames))
                #     png_bytes = graph.pipe(format='png')
                #     with open('dtree_pipe{0}.png'.format(i), 'wb') as f:
                #         f.write(png_bytes)
                # except Exception as e:
                #     print(e)

                y_est_test = dtc.predict(X_test)
                y_est_train = dtc.predict(X_train)

                df_class_train = pd.DataFrame.from_dict(
                    data={"value": pd.Series(np.asarray(y_train).squeeze()), "estimate": pd.Series(y_est_train)})
                df_class_test = pd.DataFrame.from_dict(
                    data={"value": pd.Series(np.asarray(y_test).squeeze()), "estimate": pd.Series(y_est_test)})

                Error_test[i, k] = len(df_class_test[(df_class_test['estimate'] != df_class_test['value'])]) / float(len(df_class_test))
                Error_train[i, k] = len(df_class_train[(df_class_train['estimate'] != df_class_train['value'])]) / float(len(df_class_train))

                Error_test_weighted[i, k] = score_func(df_class_test, prob_dict)
                Error_train_weighted[i, k] = score_func(df_class_train, prob_dict)

                cl_report_tst.append(classification_report(np.asarray(y_test).squeeze(), y_est_test))
                cl_report_tr.append(classification_report(np.asarray(y_train).squeeze(), y_est_train))
                conf_mat_tst.append(str(confusion_matrix(np.asarray(y_test).squeeze(), y_est_test)))
                conf_mat_tr.append(str(confusion_matrix(np.asarray(y_train).squeeze(), y_est_train)))

            k += 1
            class_report_test.append(cl_report_tst)
            class_report_train.append(cl_report_tr)
            conf_matrices_test.append(conf_mat_tst)
            conf_matrices_train.append(conf_mat_tr)

        # Visualize misclassification rate and make summary text files
        f = plt.figure(figsize=(12,8))
        plt.title(classifiers[0], fontsize=32)
        plt.boxplot(Error_test_weighted.T)
        plt.xlabel('Model complexity (max tree depth)', fontsize=28)#(min sample split)', fontsize=28)
        plt.ylabel('Test error across CV folds, K={0})'.format(K), fontsize=28)
        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(26)
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(26)
        plt.savefig("{0}_boxplot_min_sample.png".format(classifiers[0]).replace(' ', '_'))

        f = plt.figure(figsize=(12,8))
        plt.title(classifiers[0], fontsize=32)
        # plt.plot(tc, Error_train_weighted.mean(1), 'b-', linewidth=3)
        # plt.plot(tc, Error_test_weighted.mean(1), 'g-', linewidth=3)
        plt.plot(tc, Error_train.mean(1), 'r--', linewidth=3)
        plt.plot(tc, Error_test.mean(1), 'c-.', linewidth=3)
        plt.xlabel('Model complexity (max tree depth)', fontsize=28)#(min sample split)', fontsize=28)
        plt.ylabel('Misclassification rate, CV K={0}'.format(K), fontsize=28)
        plt.legend(['Training', 'Testing'], fontsize=26)
        # plt.legend(['Weighted Training', 'Weighted Testing', 'Un-Weighted Training',
        #             'Un-Weighted Testing'], fontsize=26)
        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(26)
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(26)
        plt.savefig("{0}_errors_min_sample.png".format(classifiers[0]).replace(' ', '_'))

        text_summary(class_report_test, conf_matrices_test, tc, "DECISION_TREE_TEST", "MAX TREE DEPTH", K, Error_test, Error_test_weighted, prob_dict)
        text_summary(class_report_train, conf_matrices_train, tc, "DECISION_TREE_TRAIN", "MAX TREE DEPTH", K, Error_train, Error_train_weighted, prob_dict)

    print(dt.datetime.now())
    ################################################################################################################
    ################################################################################################################


    ################################################################################################################
    ##################################            K-Nearest Neighbors             ##################################
    ################################################################################################################
    if 0:
        print("K-Nearest Neighbor Classification")
        print(dt.datetime.now())
        k=0
        # ADJUST NUMBER OF NEIGHBORS
        nn = np.arange(5, 51, 5)

        # Initialize variables
        Error_train_weighted = np.empty((len(nn), K))
        Error_test_weighted = np.empty((len(nn), K))
        Error_train = np.empty((len(nn), K))
        Error_test = np.empty((len(nn), K))

        class_report_test = []
        class_report_train = []

        conf_matrices_test = []
        conf_matrices_train = []

        # for train_index, test_index in CV.split(X, y):
        for i_index in range(K):
            # Simple holdout-set crossvalidation
            test_proportion = 0.8
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_proportion)
            print('\nComputing CV fold: {0}/{1}..'.format(k + 1, K))

            # extract training and test set for current CV fold
            # X_train, y_train = X[train_index, :], y[train_index]
            # X_test, y_test = X[test_index, :], y[test_index]

            cl_report_tst = []
            cl_report_tr = []

            conf_mat_tst = []
            conf_mat_tr = []

            # for i, n in enumerate(nn):
            for i,n in enumerate(tqdm(nn)):
                # Fit K-Nearest Neighbor classifier
                knbc = neighbors.KNeighborsClassifier(n_neighbors=n)
                print("TRAINING")
                knbc = knbc.fit(X_train, np.asarray(y_train).squeeze())
                print("TESTING TEST")
                y_est_test = knbc.predict(X_test)
                print("TESTING TRAIN")
                y_est_train = knbc.predict(X_train)


                df_class_train = pd.DataFrame.from_dict(data={"value": pd.Series(np.asarray(y_train).squeeze()), "estimate": pd.Series(y_est_train)})
                df_class_test = pd.DataFrame.from_dict(data={"value": pd.Series(np.asarray(y_test).squeeze()), "estimate": pd.Series(y_est_test)})

                #Weighted misclassification rate
                Error_test_weighted[i, k] = score_func(df_class_test, prob_dict)
                Error_train_weighted[i, k] = score_func(df_class_train, prob_dict)

                # Misclassification rate
                Error_test[i, k] = len(df_class_test[(df_class_test['estimate'] != df_class_test['value'])]) / float(len(df_class_test))
                Error_train[i, k] = len(df_class_train[(df_class_train['estimate'] != df_class_train['value'])]) / float(len(df_class_train))

                #Classification reports and confusion matrices
                cl_report_tst.append(classification_report(np.asarray(y_test).squeeze(), y_est_test))
                cl_report_tr.append(classification_report(np.asarray(y_train).squeeze(), y_est_train))
                conf_mat_tst.append(str(confusion_matrix(np.asarray(y_test).squeeze(), y_est_test)))
                conf_mat_tr.append(str(confusion_matrix(np.asarray(y_train).squeeze(), y_est_train)))

            k += 1
            class_report_test.append(cl_report_tst)
            class_report_train.append(cl_report_tr)
            conf_matrices_test.append(conf_mat_tst)
            conf_matrices_train.append(conf_mat_tr)

        # Visualize misclassification rate and make summary text file
        f = plt.figure(figsize=(12,8))
        plt.title(classifiers[1], fontsize=32)
        plt.boxplot(Error_test_weighted.T)
        plt.xlabel('Model complexity (# of neighbors)', fontsize=28)
        plt.ylabel('Test error across CV folds, K={0})'.format(K), fontsize=28)
        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(26)
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(26)
        plt.savefig("{0}_boxplot.png".format(classifiers[1]).replace(' ', '_'))

        f = plt.figure(figsize=(12,8))
        plt.title(classifiers[1], fontsize=32)
        plt.plot(nn, Error_train_weighted.mean(1), 'b-', linewidth=3)
        plt.plot(nn, Error_test_weighted.mean(1), 'g-', linewidth=3)
        plt.plot(nn, Error_train.mean(1), 'r--', linewidth=3)
        plt.plot(nn, Error_test.mean(1), 'c-.', linewidth=3)
        plt.xlabel('Model complexity (# of neighbors)', fontsize=28)
        plt.ylabel('Misclassification rate, CV K={0}'.format(K), fontsize=28)
        plt.legend(['Weighted Training', 'Weighted Testing', 'Un-Weighted Training', 'Un-Weighted Testing'], fontsize=26)
        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(26)
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(26)
        plt.savefig("{0}_errors.png".format(classifiers[1]).replace(' ', '_'))

        text_summary(class_report_test, conf_matrices_test, nn, "KNEIGHBOR_TEST", "MIN SAMPLE SPLIT", K, Error_test, Error_test_weighted, prob_dict)
        text_summary(class_report_train, conf_matrices_train, nn, "KNEIGHBOR_TREE_TRAIN", "NR OF NEIGHBORS", K, Error_train, Error_train_weighted, prob_dict)

    print(dt.datetime.now())
    ################################################################################################################
    ################################################################################################################

    ################################################################################################################
    ######################################            Naive Bayes             ######################################
    ################################################################################################################
    if 1:
        print("Naive Bayes Classification")
        print(dt.datetime.now())
        k = 0
        # ADJUST SMOOTHING FACTOR
        alpha = np.arange(0.0, 1.1, 0.1)

        # Initialize variables
        Error_train_weighted = np.empty((len(alpha), K))
        Error_test_weighted = np.empty((len(alpha), K))
        Error_train = np.empty((len(alpha), K))
        Error_test = np.empty((len(alpha), K))

        class_report_test = []  # np.empty((len(tc), K), dtype=str)
        class_report_train = []  # np.empty((len(tc), K), dtype=str)

        conf_matrices_test = []  # np.empty((len(tc), K), dtype=str)
        conf_matrices_train = []  # np.empty((len(tc), K), dtype=str)

        for train_index, test_index in CV.split(X, y):
            print('\nComputing CV fold: {0}/{1}..'.format(k + 1, K))

            # extract training and test set for current CV fold
            X_train, y_train = X[train_index, :], y[train_index]
            X_test, y_test = X[test_index, :], y[test_index]

            #Calculate priors for each class in the training set
            df_priors = pd.DataFrame.from_dict(data={"values": pd.Series(np.asarray(y_train).squeeze())})
            priors = []
            for classNr in np.sort(np.unique(df_priors['values'])):
                priors.append(len(df_priors[(df_priors['values'] == classNr)])/float(len(df_priors)))

            cl_report_tst = []
            cl_report_tr = []

            conf_mat_tst = []
            conf_mat_tr = []

            for i, a in enumerate(tqdm(alpha)):
                # Fit Naive Bayes classifier
                nbc = naive_bayes.MultinomialNB(alpha=a, fit_prior=False, class_prior=priors)
                nbc = nbc.fit(X_train, np.asarray(y_train).squeeze())

                y_est_test = nbc.predict(X_test)
                y_est_train = nbc.predict(X_train)

                df_class_train = pd.DataFrame.from_dict(
                    data={"value": pd.Series(np.asarray(y_train).squeeze()), "estimate": pd.Series(y_est_train)})
                df_class_test = pd.DataFrame.from_dict(
                    data={"value": pd.Series(np.asarray(y_test).squeeze()), "estimate": pd.Series(y_est_test)})

                #Weighted misclassification rate
                Error_test_weighted[i, k] = score_func(df_class_test, prob_dict)
                Error_train_weighted[i, k] = score_func(df_class_train, prob_dict)

                #Misclassification rate
                Error_test[i, k] = len(
                    df_class_test[(df_class_test['estimate'] != df_class_test['value'])]) / float(
                    len(df_class_test))
                Error_train[i, k] = len(
                    df_class_train[(df_class_train['estimate'] != df_class_train['value'])]) / float(
                    len(df_class_train))

                #Classification reports and confusion matrices
                cl_report_tst.append(classification_report(np.asarray(y_test).squeeze(), y_est_test))
                cl_report_tr.append(classification_report(np.asarray(y_train).squeeze(), y_est_train))
                conf_mat_tst.append(str(confusion_matrix(np.asarray(y_test).squeeze(), y_est_test)))
                conf_mat_tr.append(str(confusion_matrix(np.asarray(y_train).squeeze(), y_est_train)))

            k += 1
            class_report_test.append(cl_report_tst)
            class_report_train.append(cl_report_tr)
            conf_matrices_test.append(conf_mat_tst)
            conf_matrices_train.append(conf_mat_tr)

        #Visualize misclassification rate and make summary text file
        f = plt.figure(figsize=(12,8))
        plt.title(classifiers[2], fontsize=32)
        plt.boxplot(Error_test_weighted.T)
        plt.xlabel('Model complexity (Smoothing constant)', fontsize=28)
        plt.ylabel('Test error across CV folds, K={0})'.format(K), fontsize=28)
        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(26)
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(26)
        plt.savefig("{0}_boxplot.png".format(classifiers[2]).replace(' ', '_'))


        f = plt.figure(figsize=(12,8))
        plt.title(classifiers[2], fontsize=32)
        # plt.plot(alpha, Error_train_weighted.mean(1), 'b-', linewidth=3)
        # plt.plot(alpha, Error_test_weighted.mean(1), 'g-', linewidth=3)
        plt.plot(alpha, Error_train.mean(1), 'r--', linewidth=3)
        plt.plot(alpha, Error_test.mean(1), 'c-.', linewidth=3)
        plt.xlabel('Model complexity (Smoothing constant)', fontsize=28)
        plt.ylabel('Misclassification rate, CV K={0}'.format(K), fontsize=28)
        plt.legend(['Training', 'Testing'], fontsize=26)
        # plt.legend(['Weighted Training', 'Weighted Testing', 'Un-Weighted Training',
        #             'Un-Weighted Testing'], fontsize=26)
        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(26)
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(26)
        plt.savefig("{0}_errors.png".format(classifiers[2]).replace(' ', '_'))

        text_summary(class_report_test, conf_matrices_test, alpha, "NAIVE_BAYES_TEST", "SMOOTHING CONSTANT", K, Error_test, Error_test_weighted, prob_dict)
        text_summary(class_report_train, conf_matrices_train, alpha, "NAIVE_BAYES_TRAIN", "SMOOTHING CONSTANT", K, Error_train, Error_train_weighted, prob_dict)

    print(dt.datetime.now())
    ################################################################################################################
    ################################################################################################################

    ################################################################################################################
    ###############################            Artificial Neural Network             ###############################
    ################################################################################################################
    if 1:
        print("Artificial Neural Network Classification")
        print(dt.datetime.now())
        k = 0
        #Adjust number of nodes in hidden layers
        NHiddenUnits = np.arange(10, 101, 25)
        hidden = []
        for q in NHiddenUnits:
            for e in NHiddenUnits:
                hidden.append((q, e))

        # Initialize variables
        Error_train_weighted = np.empty((len(hidden), K))
        Error_test_weighted = np.empty((len(hidden), K))
        Error_train = np.empty((len(hidden), K))
        Error_test = np.empty((len(hidden), K))

        class_report_test = []  # np.empty((len(tc), K), dtype=str)
        class_report_train = []  # np.empty((len(tc), K), dtype=str)

        conf_matrices_test = []  # np.empty((len(tc), K), dtype=str)
        conf_matrices_train = []  # np.empty((len(tc), K), dtype=str)

        for train_index, test_index in CV.split(X, y):
            print('\nComputing CV fold: {0}/{1}..'.format(k + 1, K))

            # extract training and test set for current CV fold
            X_train, y_train = X[train_index, :], y[train_index]
            X_test, y_test = X[test_index, :], y[test_index]

            cl_report_tst = []
            cl_report_tr = []

            conf_mat_tst = []
            conf_mat_tr = []

            #Test and train for every hidden layer combination
            for i, n in enumerate(tqdm(hidden)):
                # %% Model fitting and prediction
                ## ANN Classifier, i.e. MLP with one hidden layer
                clf = neural_network.MLPClassifier(solver='adam', alpha=1e-4, activation="tanh",
                                                   hidden_layer_sizes=n, random_state=1)#, max_iter=2000)
                clf.fit(X_train, np.asarray(y_train).squeeze())

                y_est_test = clf.predict(X_test)
                y_est_train = clf.predict(X_train)

                df_class_train = pd.DataFrame.from_dict(
                    data={"value": pd.Series(np.asarray(y_train).squeeze()), "estimate": pd.Series(y_est_train)})
                df_class_test = pd.DataFrame.from_dict(
                    data={"value": pd.Series(np.asarray(y_test).squeeze()), "estimate": pd.Series(y_est_test)})

                #Weighted misclassification rate
                Error_test_weighted[i, k] = score_func(df_class_test, prob_dict)
                Error_train_weighted[i, k] = score_func(df_class_train, prob_dict)

                #Misclassification rate
                Error_test[i, k] = len(
                    df_class_test[(df_class_test['estimate'] != df_class_test['value'])]) / float(
                    len(df_class_test))
                Error_train[i, k] = len(
                    df_class_train[(df_class_train['estimate'] != df_class_train['value'])]) / float(
                    len(df_class_train))

                #Classification reports and confusion matrices
                cl_report_tst.append(classification_report(np.asarray(y_test).squeeze(), y_est_test))
                cl_report_tr.append(classification_report(np.asarray(y_train).squeeze(), y_est_train))
                conf_mat_tst.append(str(confusion_matrix(np.asarray(y_test).squeeze(), y_est_test)))
                conf_mat_tr.append(str(confusion_matrix(np.asarray(y_train).squeeze(), y_est_train)))

            k += 1
            class_report_test.append(cl_report_tst)
            class_report_train.append(cl_report_tr)
            conf_matrices_test.append(conf_mat_tst)
            conf_matrices_train.append(conf_mat_tr)

        #Visualize misclassification rate and make summary text file
        f = plt.figure(figsize=(12,8))
        plt.title(classifiers[3], fontsize=32)
        plt.boxplot(Error_test_weighted.T)
        plt.xlabel('Model complexity (# of units in hidden layers)', fontsize=28)
        plt.ylabel('Test error across CV folds, K={0})'.format(K), fontsize=28)
        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(26)
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(26)
        plt.tight_layout()
        plt.savefig("{0}_boxplot.png".format(classifiers[3]).replace(' ', '_'))

        f = plt.figure(figsize=(12,8))
        plt.title(classifiers[3], fontsize=32)
        # plt.plot(range(len(hidden)), Error_train_weighted.mean(1), 'b-', linewidth=3)
        # plt.plot(range(len(hidden)), Error_test_weighted.mean(1), 'g-', linewidth=3)
        plt.plot(range(len(hidden)), Error_train.mean(1), 'r--', linewidth=3)
        plt.plot(range(len(hidden)), Error_test.mean(1), 'c-.', linewidth=3)
        plt.xticks(range(len(hidden)), [str(x) for x in hidden], rotation=90, fontsize=26)
        plt.xlabel('Model complexity (# of units in hidden layers)', fontsize=28)
        plt.ylabel('Misclassification rate, CV K={0}'.format(K), fontsize=28)
        plt.legend(['Training', 'Testing'], fontsize=26)
        # plt.legend(['Weighted Training', 'Weighted Testing', 'Un-Weighted Training',
        #             'Un-Weighted Testing'], fontsize=26)
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(26)
        plt.tight_layout()
        plt.savefig("{0}_errors.png".format(classifiers[3]).replace(' ', '_'))

        text_summary(class_report_test, conf_matrices_test, hidden, "ANN_TEST", "HIDDEN LAYER NODES", K, Error_test, Error_test_weighted, prob_dict)
        text_summary(class_report_train, conf_matrices_train, hidden, "ANN_TRAIN", "HIDDEN LAYER NODES", K, Error_train, Error_train_weighted, prob_dict)

    print(dt.datetime.now())
    ################################################################################################################
    ################################################################################################################

    # plt.show()



def score_func(dataframe, weights):
    df_error = dataframe[(dataframe['value'] != dataframe["estimate"])]
    error_rate = 0.0
    for poker_hand in classDict.keys():
        error_rate += len(df_error[(df_error['value'] == poker_hand)])/len(dataframe) * weights[poker_hand]#len(dataframe[(dataframe['value'] == poker_hand)]) / float(len(dataframe))

    return error_rate


def main():
    print(dt.datetime.now())
    start = dt.datetime.now()
    classify()
    stop = dt.datetime.now()

    print("##################################################")
    print("##################################################")
    print("TIME ELAPSED: {0}".format((stop-start)))
    print("##################################################")
    print("##################################################")


if __name__ == '__main__':
    main()