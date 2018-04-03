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
            tc = np.arange(10, 500, 25)

        # Initialize variables
        Error_train_weighted = np.empty((len(tc), K))
        Error_test_weighted = np.empty((len(tc), K))
        Error_train = np.empty((len(tc), K))
        Error_test = np.empty((len(tc), K))

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
                dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)# min_samples_split=t)
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
        plt.savefig("{0}_boxplot_max_depth.png".format(classifiers[0]).replace(' ', '_'))

        f = plt.figure(figsize=(12,8))
        plt.title(classifiers[0], fontsize=32)
        plt.plot(tc, Error_train_weighted.mean(1), 'b-', linewidth=3)
        plt.plot(tc, Error_test_weighted.mean(1), 'g-', linewidth=3)
        plt.plot(tc, Error_train.mean(1), 'r--', linewidth=3)
        plt.plot(tc, Error_test.mean(1), 'c-.', linewidth=3)
        plt.xlabel('Model complexity (max tree depth)', fontsize=28)#(min sample split)', fontsize=28)
        plt.ylabel('Misclassification rate, CV K={0}'.format(K), fontsize=28)
        plt.legend(['Weighted Training', 'Weighted Testing', 'Un-Weighted Training',
                    'Un-Weighted Testing'], fontsize=26)
        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(26)
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(26)
        plt.savefig("{0}_errors_max_depth.png".format(classifiers[0]).replace(' ', '_'))

        text_summary(class_report_test, conf_matrices_test, tc, "DECISION_TREE_TEST_DEPTH", "MAX TREE DEPTH", K, Error_test, Error_test_weighted, prob_dict)
        text_summary(class_report_train, conf_matrices_train, tc, "DECISION_TREE_TRAIN_DEPTH", "MAX TREE DEPTH", K, Error_train, Error_train_weighted, prob_dict)

    print(dt.datetime.now())
    ################################################################################################################
    ################################################################################################################

    plt.show()



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