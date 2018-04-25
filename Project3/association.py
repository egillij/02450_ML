import numpy as np
from sklearn import model_selection
from sklearn.mixture import GaussianMixture

from Tools.toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

from Tools.writeapriorifile import WriteAprioriFile
from subprocess import run
import re
import os
import time
from sys import platform

# Global variables
classifiers = ["Decision Tree", "K-Nearest Neighbors", "Naive Bayes", "Artificial Neural Network",
               "Multinomial Logistic Regression"]

# Encode card sort and hand class in dictionary
cardSorts = {"Hearts": 1, "Spades": 2, "Diamonds": 3, "Clubs": 4}
classNames = ["Nothing", "One pair", "Two pairs", "Three of a kind", "Straight", "Flush", "Full house",
              "Four of a kind", "Straight flush", "Royal flush"]
classDict = dict(zip(range(len(classNames)), classNames))


def associate():
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
    df = pd.get_dummies(df, columns=["Hand"])

    attributeNames = df.columns

    # Extract class vector and variable matrix
    # y = df['Hand'].values
    X = df.values
    print(X.shape)

    X = X[0:100]
    # y = y[0:100000]

    print("DATA LOADED")
    print("WRITING APRIORI FILE")

    filename = os.path.join("..", "Project3", "Apriori_poker.txt")#'..{0}Data{0}courses.txt'.format(dir_sep)

    WriteAprioriFile(X, filename=filename)

    if platform.startswith('linux'):  # == "linux" or platform == "linux2":
        ext = ''  # Linux
        dir_sep = '/'
    elif platform.startswith('darwin'):  # == "darwin":
        ext = 'MAC'  # OS X
        dir_sep = '/'
    elif platform.startswith('win'):  # == "win32":
        ext = '.exe'  # Windows
        dir_sep = '\\'
    else:
        raise NotImplementedError()

    # filename = '..{0}Data{0}courses.txt'.format(dir_sep)
    minSup = 5
    minConf = 5
    maxRule = 10

    # Run Apriori Algorithm
    print('Mining for frequent itemsets by the Apriori algorithm')
    status1 = run('..{0}Tools{0}apriori{1} -f"," -s{2} -v"[Sup. %S]" {3} apriori_temp1.txt'
                  .format(dir_sep, ext, minSup, filename), shell=True)

    if status1.returncode != 0:
        print('An error occurred while calling apriori, a likely cause is that minSup was set to high such that no '
              'frequent itemsets were generated or spaces are included in the path to the apriori files.')
        exit()
    if minConf > 0:
        print('Mining for associations by the Apriori algorithm')
        status2 = run(
            '..{0}Tools{0}apriori{1} -tr -f"," -o -n{2} -c{3} -s{4} -v"[Conf. %C,Sup. %S]" {5} apriori_temp2.txt'
            .format(dir_sep, ext, maxRule, minConf, minSup, filename), shell=True)

        if status2.returncode != 0:
            print('An error occurred while calling apriori')
            exit()
    print('Apriori analysis done, extracting results')

    # Extract information from stored files apriori_temp1.txt and apriori_temp2.txt
    f = open('apriori_temp1.txt', 'r')
    lines = f.readlines()
    f.close()
    # Extract Frequent Itemsets
    FrequentItemsets = [''] * len(lines)
    sup = np.zeros((len(lines), 1))
    for i, line in enumerate(lines):
        FrequentItemsets[i] = line[0:-1]
        sup[i] = re.findall(' [-+]?\d*\.\d+|\d+]', line)[0][1:-1]
    # os.remove('apriori_temp1.txt')

    # Read the file
    f = open('apriori_temp2.txt', 'r')
    lines = f.readlines()
    f.close()
    # Extract Association rules
    AssocRules = [''] * len(lines)
    conf = np.zeros((len(lines), 1))
    for i, line in enumerate(lines):
        AssocRules[i] = line[0:-1]
        conf[i] = re.findall(' [-+]?\d*\.\d+|\d+,', line)[0][1:-1]
    # os.remove('apriori_temp2.txt')

    # sort (FrequentItemsets by support value, AssocRules by confidence value)
    AssocRulesSorted = [AssocRules[item] for item in np.argsort(conf, axis=0).ravel()]
    AssocRulesSorted.reverse()
    FrequentItemsetsSorted = [FrequentItemsets[item] for item in np.argsort(sup, axis=0).ravel()]
    FrequentItemsetsSorted.reverse()

    # Print the results
    time.sleep(.5)
    print('\n')
    print('RESULTS:\n')
    print('Frequent itemsets:')
    for i, item in enumerate(FrequentItemsetsSorted):
        print('Item: {0}'.format(item))
    print('\n')
    print('Association rules:')
    for i, item in enumerate(AssocRulesSorted):
        print('Rule: {0}'.format(item))


def main():
    print(dt.datetime.now())
    start = dt.datetime.now()
    associate()
    stop = dt.datetime.now()

    print("##################################################")
    print("##################################################")
    print("TIME ELAPSED: {0}".format((stop - start)))
    print("##################################################")
    print("##################################################")


if __name__ == '__main__':
    main()