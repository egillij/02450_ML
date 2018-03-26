# -*- coding: utf-8 -*-
import numpy as np
import xlrd
from scipy.linalg import svd
import matplotlib.pyplot as plt
from scipy.stats.mstats import zscore


# Load xls sheet with data
doc = xlrd.open_workbook('../PokerHand_all.xlsx').sheet_by_index(0)

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
    
    

# One Hot Encoder
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)
enc.fit(X)
dum = enc.transform(X)

# Assign the encoded data to X and check the Shape of the data.
X=pd.DataFrame(dum)
X.shape

# Split the data in Training and test set. We are taking 30% of samples as training data.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=.3)

# Using Decision Tree to classify the samples we get around 68% accuracy. Whihc is good for the start as we are not optimizing model.
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(x_train,y_train)
pred = clf.predict(x_test)
print(clf.score(x_test,y_test))

# Trying with Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100,random_state=0)
clf.fit(x_train,y_train)
pred = clf.predict(x_test)
print(clf.score(x_test,y_test))

from sklearn.model_selection import cross_val_score
cross_val_score(clf,X,y,cv=3)

