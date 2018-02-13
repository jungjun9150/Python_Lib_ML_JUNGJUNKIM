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

## Reading csv file
dataset_file_name = './Dataset/dataset.csv'
target_file_name = './Dataset/targetset.csv'

## Creating numpy array from csv
dataset=np.genfromtxt(dataset_file_name,delimiter=',')
target=np.genfromtxt(target_file_name,delimiter=',')
names =np.array(['InTemp','Inhumi','Otemp','Ohumi'])

## Spliting Train_data and Test_data
X_train, X_test, y_train, y_test = train_test_split(dataset,target,random_state=1)


## Drawing the Scatter matrix plot
Scatter_plot = pd.DataFrame(X_train,columns=names)
pd.plotting.scatter_matrix(Scatter_plot,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=.8,cmap=mglearn.cm3)
plt.show()

## Creating Model by K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

## Predicting the model
y_pred = knn.predict(X_test)
#print("testset correction: {:.2f}".format(np.mean(y_pred == y_test)))
print("testset correction: {:.2f}".format(knn.score(X_test,y_test)))
print("test y_pred:\n{}".format(y_pred))
print("test y_test:\n{}".format(y_test))

#print(format(dataset.shape))
#print(format(target.shape))




