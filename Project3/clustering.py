import numpy as np
from sklearn import model_selection
from sklearn.mixture import GaussianMixture

from Tools.toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats.mstats import zscore

plt.style.use("bmh")

# Global variables
classifiers = ["Decision Tree", "K-Nearest Neighbors", "Naive Bayes", "Artificial Neural Network",
               "Multinomial Logistic Regression"]

# Encode card sort and hand class in dictionary
cardSorts = {"Hearts": 1, "Spades": 2, "Diamonds": 3, "Clubs": 4}
classNames = ["Nothing", "One pair", "Two pairs", "Three of a kind", "Straight", "Flush", "Full house",
              "Four of a kind", "Straight flush", "Royal flush"]
classDict = dict(zip(range(len(classNames)), classNames))


def clustering():
    '''
    Performs a classification task on a poker hand dataset with various classification methods.
    Saves figures with errors and text files with classification summary
    :return:
    '''

    ###### LOAD ALL DATA INTO MEMORY #########################################
    print("LOADING DATA")
    df = pd.read_excel('../PokerHand_all.xlsx')
    # one out of K encoding for all categorical variables
    # df = pd.get_dummies(df, columns=['Suit 1', 'Suit 2', 'Suit 3', 'Suit 4', 'Suit 5'])
    # df = pd.get_dummies(df, columns=['Rank 1', 'Rank 2', 'Rank 3', 'Rank 4', 'Rank 5'])

    attributeNames = df.loc[:, df.columns != 'Hand'].columns

    # Extract class vector and variable matrix
    y = df['Hand'].values
    X = df.loc[:, df.columns != 'Hand'].values

    # Priors of each class in the entire dataset
    prob_dict = {}
    for hand in classDict.keys():
        prob_dict[hand] = len(df[(df['Hand'] == hand)]) / float(len(df))


    #Because of the size of the data set make a subset of the data because of the runtime
    SS = model_selection.KFold(n_splits=4, shuffle=True)
    for train_index, test_index in SS.split(X):
        X = X[test_index]
        y = y[test_index]
        break
    # X = (X - np.ones((X.shape[0],1))*X.mean(0))

    # X = X[0:25000]
    # y = y[0:25000]

    print(X.shape)


    print("DATA LOADED")

    # K-fold crossvalidation
    K = 5
    CV = model_selection.KFold(n_splits=K, shuffle=True)

    k = 0

    ####################################################################################################################
    #####################################          GAUSSIAN MIXTURE MODEL          #####################################
    ####################################################################################################################
    if 0:
        print("GMM commencing at {0}".format(dt.datetime.now()))
        # Range of K's to try
        KRange = [18, 19] #range(14, 0, -1)
        T = len(KRange)

        covar_type = 'full'  # you can try out 'diag' as well
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

            df_score = pd.DataFrame.from_dict(data={'K': KRange, 'BIC': BIC, 'AIC': AIC, 'CVE': 2 * CVE})

            writer = pd.ExcelWriter("clustering_errors.xlsx", engine="xlsxwriter")
            df_score.to_excel(writer, sheet_name="GMM")

        # Store results in excel
        df_score = pd.DataFrame.from_dict(data={'K': KRange, 'BIC': BIC, 'AIC': AIC, 'CVE': 2*CVE})

        writer = pd.ExcelWriter("clustering_errors.xlsx", engine="xlsxwriter")
        df_score.to_excel(writer, sheet_name="GMM")

        # Plot results

        plt.figure(figsize=(12, 8))
        plt.plot(KRange[:10], BIC[:10], '-*b', linewidth=3, markersize=10)
        plt.plot(KRange[:10], AIC[:10], '-xr', linewidth=3, markersize=10)
        plt.plot(KRange[:10], 2 * CVE[:10], '-ok', linewidth=3, markersize=10)
        plt.legend(['BIC', 'AIC', 'Crossvalidation'], fontsize=26)
        plt.xlabel('Number of components (K)', fontsize=28)
        plt.title('GMM Model validation', fontsize=32)
        plt.rc('font', **{'size': '26'})
        # plt.ticklabel_format(useOffset=False, style='sci', axis='y', size=26)
        for tick in plt.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(26)

        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(26)
        plt.gca().yaxis.offsetText.set_fontsize(26)

        plt.savefig("GMM_likelihoods_max10.png")

        print("GMM completed at {0}".format(dt.datetime.now()))

    if 1:
        print("GMM commencing at {0}".format(dt.datetime.now()))
        # Range of K's to try

        covar_type = 'full'  # you can try out 'diag' as well
        reps = 10  # number of fits with different initalizations, best result will be kept

        # K-fold crossvalidation
        CV = model_selection.KFold(n_splits=10, shuffle=True)
        K=17
        print('Fitting model for K={0}'.format(K))

        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X)

        gmm_mean = gmm.means_
        gmm_cov = gmm.covariances_
        gmm_weights = gmm.weights_

        y_all_test = []
        cls_all_test = []

        for train_index, test_index in CV.split(X):
            # extract training and test set for current CV fold
            X_train = X[train_index]
            X_test = X[test_index]
            y_test = y[test_index]

            # Fit Gaussian mixture model to X_train
            gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

            y_all_test.extend(y_test)
            cls_gmm = gmm.predict(X_test)
            cls_all_test.extend(cls_gmm)

        # print(y_all_test)
        # print(cls_all_test)

        # print("\nMEANS:\n{0}".format(str(gmm_mean)))
        # print("\n\nCOVARIANCE:\n{0}".format(str(gmm_cov)))
        # print("\n\nWEIGHTS:\n{0}".format(str(gmm_weights)))

        with open("GMM_{0}_clusters.txt".format(K), 'w') as fw:
            fw.write('\nMEANS:\n{0}'.format(str(gmm_mean)))
            fw.write("\n\nCOVARIANCE:\n{0}".format(str(gmm_cov)))
            fw.write("\n\nWEIGHTS:\n{0}".format(str(gmm_weights)))

        gmm_distribution = validate_clusters(cls_all_test, y_all_test)

        for i in range(len(gmm_distribution)):
            plt.figure(figsize=(15, 10))
            hist, bins = np.histogram(gmm_distribution[i], bins=[i for i in range(11)], normed=True)
            print(hist)
            print(bins)
            plt.bar(bins[0:-1], hist*100.0, align="center")
            plt.title("Normalized Histogram\nCLUSTER " + str(i + 1), fontsize=44)
            plt.xlabel("CLASS LABEL", fontsize=40)
            plt.ylabel("PERCENTAGE OF CLASS [%]", fontsize=40)
            plt.grid()
            for tick in plt.gca().xaxis.get_major_ticks():
                tick.label.set_fontsize(38)

            for tick in plt.gca().yaxis.get_major_ticks():
                tick.label.set_fontsize(38)
            plt.savefig("GMM_CLUSTER_{0}_histogram.png".format(i + 1))

        print("GMM completed at {0}".format(dt.datetime.now()))



    ####################################################################################################################
    #####################################          HIERARCHICAL CLUSTERING          ####################################
    ####################################################################################################################
    if 0:
        # Perform hierarchical/agglomerative clustering on data matrix
        # Method = 'single'
        # Method = 'complete'
        # Method = 'average'
        Method = 'ward'
        Metric = 'euclidean'

        Z = linkage(X, method=Method, metric=Metric)

        # Compute and display clusters by thresholding the dendrogram
        Maxclust = 17
        cls = fcluster(Z, criterion='maxclust', t=Maxclust)
        plt.figure()
        clusterplot(X, cls, y=y)

        # Display dendrogram
        max_display_levels = 14
        plt.figure(figsize=(10, 4))
        dendrogram(Z, truncate_mode='level', p=max_display_levels)
        plt.savefig("dendogram.png")

        cluster_distribution = validate_clusters(cls, y)
        for i in range(len(cluster_distribution)):
            plt.figure(figsize=(15, 10))
            # print(cluster_distribution[i])
            hist, bins = np.histogram(cluster_distribution[i], bins=[i for i in range(11)], normed=True)
            plt.bar(bins[0:-1], hist * 100.0, align="center")
            # plt.hist(cluster_distribution[i], normed=True)
            plt.title("Normalized Histogram\nCLUSTER " + str(i+1), fontsize=44)
            plt.xlabel("CLASS LABEL", fontsize=40)
            plt.ylabel("PERCENTAGE OF CLASS [%]", fontsize=40)
            plt.grid()
            for tick in plt.gca().xaxis.get_major_ticks():
                tick.label.set_fontsize(38)

            for tick in plt.gca().yaxis.get_major_ticks():
                tick.label.set_fontsize(38)

            plt.savefig("CLUSTER_{0}_histogram.png".format(i+1))

    # plt.show()

def validate_clusters(clusters, classes):
    nr_clusters = len(np.unique(clusters))
    nr_classes = len(np.unique(clusters))

    data = [[] for x in range(nr_clusters)]
    for i_ind in range(len(clusters)):
        data[clusters[i_ind]-1].append(classes[i_ind])

    return data

def main():
    print(dt.datetime.now())
    start = dt.datetime.now()
    clustering()
    stop = dt.datetime.now()

    print("##################################################")
    print("##################################################")
    print("TIME ELAPSED: {0}".format((stop - start)))
    print("##################################################")
    print("##################################################")


if __name__ == '__main__':
    main()