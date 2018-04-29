from tqdm import tqdm
import numpy as np
from sklearn import model_selection
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing

from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors

import pandas as pd
from matplotlib.pyplot import figure, bar, title, plot, show, subplot, imshow, xticks, yticks, cm
import datetime as dt

# Encode card sort and hand class in dictionary
cardSorts = {"Hearts": 1, "Spades": 2, "Diamonds": 3, "Clubs": 4}
classNames = ["Nothing", "One pair", "Two pairs", "Three of a kind", "Straight", "Flush", "Full house",
              "Four of a kind", "Straight flush", "Royal flush"]
classDict = dict(zip(range(len(classNames)), classNames))


def findOutliers():
    ###### LOAD ALL DATA INTO MEMORY #########################################
    print("LOADING DATA")
    df = pd.read_excel('../PokerHand_all.xlsx')

    X = df.iloc[:, :-1].values
    
    #Because of the size of the data set make a subset of the data because of the runtime
    SS = model_selection.KFold(n_splits=40, shuffle=True)
    for train_index, test_index in SS.split(X):
        X = X[test_index]
        break
    
    # Normalize the data
    X = (X - np.ones((X.shape[0],1))*X.mean(0))

    print("DATA LOADED")
    
    # Neighbor to use:
    K = 5

    if 1:
        # Estimate the optimal kernel density width, by leave-one-out cross-validation
        
        widths = X.var(axis=0).max() * (2.0**np.arange(-10,3))
        logP = np.zeros(np.size(widths))
        for i,w in enumerate(widths):
           density, log_density = gausKernelDensity(X,w)
           logP[i] = log_density.sum()
           
        val = logP.max()
        ind = logP.argmax()
        
        width=widths[ind]
        
        print('logP.max:')
        print(val)
        
        print('logP.argmax:')
        print(ind)
        
        print('Width array:')
        print(widths)
        
        print('Optimal estimated width is: {0}'.format(width))
        
        # evaluate density for estimated width
        density, log_density = gausKernelDensity(X,width)
        
        # Sort the densities
        i = (density.argsort(axis=0)).ravel()
        density = density[i].reshape(-1,)
        
        # Display the index of the lowest density data object
        print('Lowest density: {0} for data object: {1}'.format(density[0],i[0]))
        print('Lowest density: {0} for data object: {1}'.format(density[1],i[1]))
        print('Lowest density: {0} for data object: {1}'.format(density[2],i[2]))
        print('Lowest density: {0} for data object: {1}'.format(density[3],i[3]))
        print('Lowest density: {0} for data object: {1}'.format(density[4],i[4]))
        
        # Plot density estimate of outlier score
        figure(1)
        bar(range(20),density[:20])
        title('Density estimate')
        figure(2)
        plot(logP)
        title('Optimal width')
        show()
               
    if 1:
        ### K-neighbors density estimator        
        
        # Find the k nearest neighbors
        knn = NearestNeighbors(n_neighbors=K).fit(X)
        D, i = knn.kneighbors(X)
        
        density = 1./(D.sum(axis=1)/K)
        
        # Sort the scores
        i = density.argsort()
        density = density[i]
        
        # Display the index of the lowest density data object
        print('Lowest density: {0} for data object: {1}'.format(density[0],i[0]))
        print('Lowest density: {0} for data object: {1}'.format(density[1],i[1]))
        print('Lowest density: {0} for data object: {1}'.format(density[2],i[2]))
        print('Lowest density: {0} for data object: {1}'.format(density[3],i[3]))
        print('Lowest density: {0} for data object: {1}'.format(density[4],i[4]))
        
        # Plot k-neighbor estimate of outlier score (distances)
        figure(3)
        bar(range(20),density[:20])
        title('KNN density: Outlier score')
        show()
        
    if 1:
        ### K-nearest neigbor average relative density
        # Compute the average relative density
        
        knn = NearestNeighbors(n_neighbors=K).fit(X)
        D, i = knn.kneighbors(X)
        density = 1./(D.sum(axis=1)/K)
        avg_rel_density = density/(density[i[:,1:]].sum(axis=1)/K)
        
        # Sort the avg.rel.densities
        i_avg_rel = avg_rel_density.argsort()
        avg_rel_density = avg_rel_density[i_avg_rel]
        
        # Display the index of the lowest density data object
        print('Lowest density: {0} for data object: {1}'.format(avg_rel_density[0],i_avg_rel[0]))
        print('Lowest density: {0} for data object: {1}'.format(avg_rel_density[1],i_avg_rel[1]))
        print('Lowest density: {0} for data object: {1}'.format(avg_rel_density[2],i_avg_rel[2]))
        print('Lowest density: {0} for data object: {1}'.format(avg_rel_density[3],i_avg_rel[3]))
        print('Lowest density: {0} for data object: {1}'.format(avg_rel_density[4],i_avg_rel[4]))
        
        # Plot k-neighbor estimate of outlier score (distances)
        figure(5)
        bar(range(20),avg_rel_density[:20])
        title('KNN average relative density: Outlier score')
        show()

def main():
    print(dt.datetime.now())
    start = dt.datetime.now()
    findOutliers()
    stop = dt.datetime.now()

    print("##################################################")
    print("##################################################")
    print("TIME ELAPSED: {0}".format((stop - start)))
    print("##################################################")
    print("##################################################")


if __name__ == '__main__':
    main()