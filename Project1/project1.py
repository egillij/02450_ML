# exercise 2.1.1
import numpy as np
import xlrd
from scipy.linalg import svd
import matplotlib.pyplot as plt
from scipy.stats.mstats import zscore


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
    # Load xls sheet with data
    doc = xlrd.open_workbook('../PokerHand_Training.xlsx').sheet_by_index(0)

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

    # Calculate summary statistics and print them to console
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


    # Standardize data by removing the mean
    X_hat =  X - np.ones((N-1,1))*X.mean(0) #zscore(X)

    #Outlier detection, boxplot and histograms
    plt.figure()
    plt.title('Poker Hand: Boxplot original', fontsize=18)
    plt.boxplot(X, attributeNames)
    plt.xticks(range(1, M + 1), attributeNames, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("attribute_boxplots.png", bbox_inches="tight")

    plt.figure(figsize=(14, 9))
    plt.suptitle('Poker hand - Histograms', fontsize=18)
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
        plt.title(attributeNames[i], fontsize=16)
        plt.xticks(range(1, binnumber+1))  # yticks([])
        plt.ylim(0, N)  # Make the y-axes equal for improved readability
        plt.grid()
        ax.get_xaxis().grid(False)
        if i % v != 0: ax.get_yaxis().set_ticklabels([])#yticks([])

    plt.savefig("attribute_histograms.png", bbox_inches="tight")


    # Correlation matrix to image
    plt.figure()
    plt.imshow(corr_x)
    plt.colorbar()
    plt.set_cmap(cmap="Greys")
    plt.title("Correlation matrix", fontsize=18)
    plt.xticks(range(0, M), attributeNames, rotation=45, fontsize=14)
    plt.yticks(range(0, M), attributeNames, fontsize=14)
    plt.savefig("correlation_matrix.png", bbox_inches="tight")


    # Principal component analysis
    # PCA by computing SVD of Y
    U,S,V = svd(X_hat, full_matrices=False)
    V = V.T
    Vmat = np.mat(V)
    for icol in range(M):
        print('PC{0}'.format(icol+1))
        mat2latex(Vmat[:, icol].T)
        print()


    # Project the centered data onto principal component space
    Z = X_hat * V

    #Plot original data as an image
    plt.figure()
    plt.imshow(X_hat, cmap="seismic", aspect="auto")
    plt.colorbar()
    plt.xticks(range(0, M), attributeNames, rotation=45, fontsize=14)
    plt.title("Data matrix with zero mean", fontsize=18)
    plt.savefig("zero_mean_data.png", bbox_inches="tight")

    #Plot projected data as an image
    plt.figure()
    plt.imshow(Z, cmap="seismic", aspect="auto")
    plt.colorbar()
    plt.xticks(range(0, M), ["PC {0}".format(i_index+1) for i_index in range(0, M)], rotation=45, fontsize=14)
    plt.title("Data matrix projected onto PCA", fontsize=18)
    plt.savefig("PCA_projected_data.png", bbox_inches="tight")
    # plt.show()

    # Compute variance explained by principal components
    rho = (S*S) / (S*S).sum()
    print("PCA variance")
    print(rho)
    print("PCA cumulative variance")
    print(np.cumsum(rho))

    # Plot variance explained by the PC's
    plt.figure()
    plt.plot(range(1,len(rho)+1),rho*100.0,'o-')
    plt.xticks(range(1, len(rho) + 1), fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Variance explained by principal components', fontsize=18)
    plt.xlabel('Principal component', fontsize=16)
    plt.ylabel('Variance explained [%]', fontsize=16)
    plt.grid()
    plt.savefig("PCA_var.png", bbox_inches="tight")
    #Plot the cumulative variance explained by the PC's
    plt.figure()
    plt.plot(range(1,len(rho)+1), np.cumsum(rho)*100.0,'o-')
    plt.xticks(range(1, len(rho) + 1), fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Variance explained by principal components', fontsize=18)
    plt.xlabel('Principal component', fontsize=16)
    plt.ylabel('Variance explained [%]', fontsize=16)
    plt.grid()
    plt.savefig("PCA_var_cum.png", bbox_inches="tight")

    #Plot the principal component matrix as an image
    plt.figure()
    plt.imshow(V, cmap="seismic", aspect="auto")
    plt.title("Principal Component Analysis", fontsize=18)
    plt.xticks(range(0, M), ["PC {0}".format(i_index + 1) for i_index in range(0, M)], rotation=45, fontsize=14)
    plt.yticks(range(0, M), attributeNames, fontsize=14)
    plt.colorbar()
    plt.savefig("PCA_image.png", bbox_inches="tight")

    # ' Make a matrix plot of every principal component comparison
    # TODO: Fix the axes if the figure is useful.
    # plt.figure(figsize=(14,9))
    # plt.title('PCA')
    # for i in range(M):
    #     for j in range(M):
    #         # i=0
    #         # j=1[
    #         # Plot PCA of the data
    #         plt.subplot(M, M, i * M + j + 1)
    #         # Z = array(Z)
    #         for c in range(C):
    #             # select indices belonging to class c:
    #             class_mask = y.A.ravel() == c
    #             plt.plot(Z[class_mask, i], Z[class_mask, j], 'o')
    #             plt.legend(classNames)
    #             plt.xlabel('PC{0}'.format(i + 1))
    #             plt.ylabel('PC{0}'.format(j + 1))
    #
    # plt.savefig("PC_comparisons.png")


    #Plot the PC projected data in parallel coordinates. One graph for each poker hand class.
    plt.figure()
    #Predefined colors for the classes
    cols = ['#000000', '#a00909', '#ff4747', '#f600ff', '#541df7', '#6dccff', '#00ff99', '#0cba00', '#d8bb00', '#ffa921']#colmap([o*30 for o in range(len(classNames))])#[colmap(o) for o in range(len(classNames))]

    #Extract each class into a vector of their own
    parallel_coord = [[] for x in range(len(classNames))]
    for ind in range(N-1):
        parallel_coord[int(y.T.tolist()[0][ind])].append(Z[ind][:].tolist()[0])

    #Plot each class on a separate subfigure
    plt.figure(figsize=(14, 9))
    plt.suptitle('Data projected onto PCA - parallel coordinates', fontsize=18)
    u = np.floor(np.sqrt(len(classNames)))
    v = np.ceil(float(len(classNames)) / u)
    for i in range(C):
        ax = plt.subplot(u, v, i + 1)
        for ys in parallel_coord[i]:
            ax.plot(range(1,M+1), ys, c=cols[i])
        plt.title(classNames[i], fontsize=16)
        plt.xticks(range(1, M + 1))  # yticks([])
        plt.ylim(-13, 13)  # Make the y-axes equal for improved readability
        plt.grid()
        if i % v != 0: ax.get_yaxis().set_ticklabels([])  # yticks([])
        else: plt.ylabel("Coordinate value")

    plt.savefig("data_proj_pca_parallel.png")



    # Output result to screen
    # plt.show()


if __name__ == '__main__':
    main()