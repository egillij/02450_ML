# exercise 2.1.1
from tqdm import tqdm
import numpy as np
from sklearn import model_selection, tree, naive_bayes, neighbors, neural_network, linear_model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import graphviz
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from scipy import stats

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

        inner_K = 5

        # Initialize variables
        Error_train_weighted_dectree = np.empty((inner_K, K))
        Error_test_weighted_dectree = np.empty((inner_K, K))
        Error_train_dectree = np.empty((inner_K, K))
        Error_test_dectree = np.empty((inner_K, K))

        for train_index, test_index in CV.split(X, y):
            print('Computing CV fold: {0}/{1}..'.format(k + 1, K))

            # extract training and test set for current CV fold
            X_train, y_train = X[train_index, :], y[train_index]
            # X_test, y_test = X[test_index, :], y[test_index]

            k_i = 0
            CV2 = model_selection.KFold(n_splits=inner_K, shuffle=True)
            # Test and train for every hidden layer combination
            for train_index_inner, test_index_inner in tqdm(CV2.split(X_train, y_train)):
                X_train_inner, y_train_inner = X_train[train_index_inner, :], y_train[train_index_inner]
                X_test_inner, y_test_inner = X_train[test_index_inner, :], y_train[test_index_inner]
                dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=40)# max_depth=t
                dtc = dtc.fit(X_train_inner, y_train_inner)

                # try:
                #     #Visualize decision tree
                #     graph = graphviz.Source(tree.export_graphviz(dtc, out_file=None, feature_names=attributeNames))
                #     png_bytes = graph.pipe(format='png')
                #     with open('dtree_pipe{0}.png'.format(i), 'wb') as f:
                #         f.write(png_bytes)
                # except Exception as e:
                #     print(e)

                y_est_test = dtc.predict(X_test_inner)
                y_est_train = dtc.predict(X_train_inner)

                df_class_train = pd.DataFrame.from_dict(
                    data={"value": pd.Series(np.asarray(y_train_inner).squeeze()), "estimate": pd.Series(y_est_train)})
                df_class_test = pd.DataFrame.from_dict(
                    data={"value": pd.Series(np.asarray(y_test_inner).squeeze()), "estimate": pd.Series(y_est_test)})

                Error_test_dectree[k_i, k] = len(df_class_test[(df_class_test['estimate'] != df_class_test['value'])]) / float(len(df_class_test))
                Error_train_dectree[k_i, k] = len(df_class_train[(df_class_train['estimate'] != df_class_train['value'])]) / float(len(df_class_train))

                Error_test_weighted_dectree[k_i, k] = score_func(df_class_test, prob_dict)
                Error_train_weighted_dectree[k_i, k] = score_func(df_class_train, prob_dict)

                k_i += 1

            k += 1

        print('TESTING ERROR\n {0}'.format(str(Error_test_dectree)))
        print('TRAINING ERROR\n {0}'.format(str(Error_train_dectree)))
        cred_interval(Error_test_dectree, "Decision Tree")

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
        inner_K = 5

        # Initialize variables
        Error_train_weighted_nb = np.empty((inner_K, K))
        Error_test_weighted_nb = np.empty((inner_K, K))
        Error_train_nb = np.empty((inner_K, K))
        Error_test_nb = np.empty((inner_K, K))



        for train_index, test_index in CV.split(X, y):
            print('\nComputing CV fold: {0}/{1}..'.format(k + 1, K))

            # extract training and test set for current CV fold
            X_train, y_train = X[train_index, :], y[train_index]

            k_i = 0
            CV2 = model_selection.KFold(n_splits=inner_K, shuffle=True)
            # Test and train for every hidden layer combination
            for train_index_inner, test_index_inner in tqdm(CV2.split(X_train, y_train)):
                X_train_inner, y_train_inner = X_train[train_index_inner, :], y_train[train_index_inner]
                X_test_inner, y_test_inner = X_train[test_index_inner, :], y_train[test_index_inner]

                # Calculate priors for each class in the training set
                df_priors = pd.DataFrame.from_dict(data={"values": pd.Series(np.asarray(y_train).squeeze())})
                priors = []
                for classNr in np.sort(np.unique(df_priors['values'])):
                    priors.append(len(df_priors[(df_priors['values'] == classNr)]) / float(len(df_priors)))

                # Fit Naive Bayes classifier
                nbc = naive_bayes.MultinomialNB(alpha=1.0, fit_prior=False, class_prior=priors)
                nbc = nbc.fit(X_train_inner, np.asarray(y_train_inner).squeeze())

                y_est_test = nbc.predict(X_test_inner)
                y_est_train = nbc.predict(X_train_inner)

                df_class_train = pd.DataFrame.from_dict(
                    data={"value": pd.Series(np.asarray(y_train_inner).squeeze()), "estimate": pd.Series(y_est_train)})
                df_class_test = pd.DataFrame.from_dict(
                    data={"value": pd.Series(np.asarray(y_test_inner).squeeze()), "estimate": pd.Series(y_est_test)})

                #Weighted misclassification rate
                Error_test_weighted_nb[k_i, k] = score_func(df_class_test, prob_dict)
                Error_train_weighted_nb[k_i, k] = score_func(df_class_train, prob_dict)

                #Misclassification rate
                Error_test_nb[k_i, k] = len(
                    df_class_test[(df_class_test['estimate'] != df_class_test['value'])]) / float(
                    len(df_class_test))
                Error_train_nb[k_i, k] = len(
                    df_class_train[(df_class_train['estimate'] != df_class_train['value'])]) / float(
                    len(df_class_train))

                k_i += 1

            k += 1

        print('TESTING ERROR\n {0}'.format(str(Error_test_nb)))
        print('TRAINING ERROR\n {0}'.format(str(Error_train_nb)))
        cred_interval(Error_test_nb, "Naive Bayes")


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

        inner_K = 5

        # Initialize variables
        Error_train_weighted_ann = np.empty((inner_K, K))
        Error_test_weighted_ann = np.empty((inner_K, K))
        Error_train_ann = np.empty((inner_K, K))
        Error_test_ann = np.empty((inner_K, K))

        for train_index, test_index in CV.split(X, y):
            print('\nComputing CV fold: {0}/{1}..'.format(k + 1, K))

            # extract training and test set for current CV fold
            X_train, y_train = X[train_index, :], y[train_index]
            # X_test, y_test = X[test_index, :], y[test_index]


            k_i = 0
            CV2 = model_selection.KFold(n_splits=inner_K, shuffle=True)
            #Test and train for every hidden layer combination
            for train_index_inner, test_index_inner in tqdm(CV2.split(X_train, y_train)):
                # print(train_index_inner)
                # print(test_index_inner)
                X_train_inner, y_train_inner = X_train[train_index_inner, :], y_train[train_index_inner]
                X_test_inner, y_test_inner = X_train[test_index_inner, :], y_train[test_index_inner]

                # %% Model fitting and prediction
                ## ANN Classifier, i.e. MLP with one hidden layer
                clf = neural_network.MLPClassifier(solver='adam', alpha=1e-4, activation="tanh",
                                                   hidden_layer_sizes=(60, 60), random_state=1)#, max_iter=2000)
                clf.fit(X_train_inner, np.asarray(y_train_inner).squeeze())

                y_est_test = clf.predict(X_test_inner)
                y_est_train = clf.predict(X_train_inner)

                df_class_train = pd.DataFrame.from_dict(
                    data={"value": pd.Series(np.asarray(y_train_inner).squeeze()), "estimate": pd.Series(y_est_train)})
                df_class_test = pd.DataFrame.from_dict(
                    data={"value": pd.Series(np.asarray(y_test_inner).squeeze()), "estimate": pd.Series(y_est_test)})

                #Weighted misclassification rate
                Error_test_weighted_ann[k_i, k] = score_func(df_class_test, prob_dict)
                Error_train_weighted_ann[k_i, k] = score_func(df_class_train, prob_dict)

                #Misclassification rate
                Error_test_ann[k_i, k] = len(
                    df_class_test[(df_class_test['estimate'] != df_class_test['value'])]) / float(
                    len(df_class_test))
                Error_train_ann[k_i, k] = len(
                    df_class_train[(df_class_train['estimate'] != df_class_train['value'])]) / float(
                    len(df_class_train))

                k_i += 1

            k += 1

        print('TESTING ERROR\n {0}'.format(str(Error_test_ann)))
        print('TRAINING ERROR\n {0}'.format(str(Error_train_ann)))
        cred_interval(Error_test_ann, "Artificial Neural Network")

    print(dt.datetime.now())
    ################################################################################################################
    ################################################################################################################

    # plt.show()

    z = (Error_test_dectree.mean(1) - Error_test_ann.mean(1))
    zb = z.mean()
    nu = K - 1
    sig = (z - zb).std() / np.sqrt(K - 1)
    alpha = 0.05
    zl_1 = sig * stats.t.ppf(alpha / 2, nu)
    zL = zb + sig * zl_1
    zh_1 = sig * stats.t.ppf(1 - alpha / 2, nu)
    zH = zb + zh_1

    with open('dectree_ann_comparison.txt', 'w') as ofile:
        ofile.write('COMPARISON STATISTICS\n\n')
        ofile.write('ERROR COMPARISON: {0}\n'.format(str(z)))
        ofile.write('MEAN ERROR: {0}\n'.format(zb))
        ofile.write('SIGMA: {0}\n'.format(sig))
        ofile.write('alpha: {0}\n'.format(alpha))
        ofile.write('zL addition: {0}\n'.format(zl_1))
        ofile.write('zH addition: {0}\n'.format(zh_1))
        ofile.write('zL: {0}\n'.format(zL))
        ofile.write('zH: {0}\n'.format(zH))
        if zL <= 0 and zH >= 0:
            ofile.write('Classifiers are not significantly different')
        else:
            ofile.write('Classifiers are significantly different.')


def cred_interval(error_list, method_name):
    accuracy = (1.0 - error_list).mean(1)
    mean_acc = accuracy.mean()
    nu = len(mean_acc)-1
    sig = (accuracy-mean_acc).std() / np.sqrt(nu)
    alpha = 0.05
    accL = mean_acc + stats.t.ppf(alpha / 2, nu)
    accH = mean_acc + stats.t.ppf(1 - alpha/2, nu)

    with open('{0}_stats.txt'.format(method_name.replace(' ', '_'))) as of:
        of.write('Accuracy: {0}\n'.format(str(accuracy)))
        of.write('Mean Accuracy: {0}\n'.format(str(mean_acc)))
        of.write('Sigma: {0}\n'.format(str(sig)))
        of.write('Left: {0}\n'.format(str(accL)))
        of.write('Right: {0}\n'.format(str(accH)))


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