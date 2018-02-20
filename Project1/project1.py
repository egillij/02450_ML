# exercise 2.1.1
import numpy as np
import xlrd
from scipy.linalg import svd
import matplotlib.pyplot as plt


def mat2latex(X):

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
    # Load xls sheet with data
    doc = xlrd.open_workbook('../PokerHand_Training.xlsx').sheet_by_index(0)

    # Encode card sort and hand class in dictionary
    cardSorts = {"Hearts": 1, "Spades": 2, "Diamonds": 3, "Clubs": 4}
    classNames = ["Nothing", "One pair", "Two pairs", "Three of a kind", "Straight", "Flush", "Full house",
                  "Four of a kind", "Straight flush", "Royal flush"]
    classDict = dict(zip(classNames, range(5)))

    # Compute values of N, M and C.
    N = doc.nrows - 1  # number of rows in the dataset
    M = doc.ncols - 1  # number of columns in dataset
    C = len(classNames)

    attributeNames = doc.row_values(0, 0, M)
    print(attributeNames)


    # Extract vector y, convert to NumPy matrix and transpose
    y = np.mat(doc.col_values(doc.ncols - 1, 1, N)).T

    # Preallocate memory, then extract excel data to matrix X
    X = np.mat(np.empty((N - 1, M)))
    for i, col_id in enumerate(range(0, M)):
        X[:, i] = np.mat(doc.col_values(col_id, 1, N)).T

    # Summary statistics
    mean_x = X.mean(axis=0)
    std_x = X.std(ddof=1, axis=0)
    median_x = np.median(X, axis=0)
    range_x = X.max(axis=0) - X.min(axis=0)

    var_x = np.multiply(std_x, std_x)
    cov_x = np.cov(X, rowvar=False)
    corr_x = np.corrcoef(X, rowvar=False)


    print("Mean")
    mat2latex(mean_x)

    print("Stdev")
    mat2latex(std_x)

    print("Median")
    mat2latex(median_x)

    print("Range")
    mat2latex(range_x)

    print("Variance")
    mat2latex(var_x)

    print("Covariance")
    mat2latex(cov_x)

    print("Correlation")
    mat2latex(corr_x)
    #
    # print()

    # Remove mean from data
    Y = X - np.ones((N-1,1))*X.mean(0)

    #Outlier detection
    plt.figure()
    plt.title('Poker Hand: Boxplot original')
    plt.boxplot(X, attributeNames)

    plt.xticks(range(1, M + 1), attributeNames, rotation=45)

    plt.figure(figsize=(14, 9))
    u = np.floor(np.sqrt(M))
    v = np.ceil(float(M) / u)
    for i in range(M):
        ax = plt.subplot(u, v, i + 1)
        if "Rank" in attributeNames[i]:
            binnumber = 13
        else:
            binnumber = 4
        hist, bins = np.histogram(X[:, i], bins=binnumber)
        plt.bar(range(1, binnumber+1), hist)
        plt.xlabel(attributeNames[i])
        ax.get_xaxis().set_ticks(range(1, binnumber+1))  # yticks([])
        plt.ylim(0, N)  # Make the y-axes equal for improved readability
        plt.grid()
        ax.get_xaxis().grid(False)
        if i % v != 0: ax.get_yaxis().set_ticklabels([])#yticks([])
        if i == 0: plt.title('Poker hand - Histograms')

    plt.show()

    # Principal component analysis
    # PCA by computing SVD of Y
    U,S,V = svd(Y, full_matrices=False)
    V = V.T
    Vmat = np.mat(V)
    for icol in range(M):
        print('PC{0}'.format(icol+1))
        mat2latex(Vmat[:, icol].T)
        print()


    # Project the centered data onto principal component space
    Z = Y * V


    # Compute variance explained by principal components
    rho = (S*S) / (S*S).sum()
    print("PCA variance")
    print(rho)
    # Plot variance explained
    plt.figure()
    plt.plot(range(1,len(rho)+1),rho*100.0,'o-')
    plt.gca().get_xaxis().set_ticks(range(1, len(rho) + 1))
    plt.title('Variance explained by principal components', fontsize=18)
    plt.xlabel('Principal component', fontsize=16)
    plt.ylabel('Variance explained [%]', fontsize=16)
    plt.grid()
    plt.savefig("PCA_var.png")

    plt.figure()
    plt.plot(range(1,len(rho)+1), np.cumsum(rho)*100.0,'o-')
    plt.gca().get_xaxis().set_ticks(range(1, len(rho) + 1))
    plt.title('Variance explained by principal components', fontsize=18)
    plt.xlabel('Principal component', fontsize=16)
    plt.ylabel('Variance explained [%]', fontsize=16)
    plt.grid()
    plt.savefig("PCA_var_cum.png")
    plt.show()

    # Make a matrix plot of every principal component comparison
    # TODO: Fix the axes.
    # f = plt.figure()
    # plt.title('PCA')
    # for i in range(M):
    #     for j in range(M):
    #         # i=0
    #         # j=1
    #         # Plot PCA of the data
    #         plt.subplot(M,M, i*M+j+1)
    #         #Z = array(Z)
    #         for c in range(C):
    #             # select indices belonging to class c:
    #             class_mask = y.A.ravel()==c
    #             plt.plot(Z[class_mask,i], Z[class_mask,j], 'o')
    #             # plt.legend(classNames)
    #             plt.xlabel('PC{0}'.format(i+1))
    #             plt.ylabel('PC{0}'.format(j+1))
    #
    # # Output result to screen
    # plt.show()


if __name__ == '__main__':
    main()