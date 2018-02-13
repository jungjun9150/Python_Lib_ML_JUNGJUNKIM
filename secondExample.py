import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import xlrd
from IPython.display import clear_output, Image, display
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pandas import ExcelWriter
from pandas import ExcelFile
from openpyxl import load_workbook
from numpy import genfromtxt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# example 1
"""
X,y = mglearn.datasets.make_wave(n_samples=100)
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel("attr")
plt.ylabel("target")
plt.show()
"""

# example 2
"""
boston = load_boston()
X,y = mglearn.datasets.load_extended_boston()
"""

# example 3
"""
mglearn.plots.plot_knn_classification(n_neighbors=5)
plt.show()
"""

# example 4
"""
X,y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)

print("testset correction:{:.2f}".format(clf.score(X_test,y_test)))
"""

# example 5 

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []
neighbors_settings = range(1,11)
