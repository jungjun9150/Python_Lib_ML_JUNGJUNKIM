import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import clear_output, Image, display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()

print("target names: {}".format(iris_dataset['target_names'])) ## ['setoa' 'versicolor' 'virginica']
print("0~10's data:\n{}".format(iris_dataset['data'][:10]))    ## Sample data 0~10's
print("target:\n{}".format(iris_dataset['target']))            ## Target 100's

Input_train,Input_test,output_train,output_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0) ## Split the data to (train_data : test_data) as (75% : 25%)

## Visually confirm the Data ##
#print("Input train data:{}".format(Input_train))
#print("output train data:{}".format(output_train))
#print("Input test data:{}".format(Input_test))
#print("output test data:{}".format(output_test))

## Check the Data size
#print("Input train data size:{}".format(Input_train.shape))
#print("output train data size:{}".format(output_train.shape))
#print("Input test data size:{}".format(Input_test.shape))
#print("output test data size:{}".format(output_test.shape))

## Visualizing Scatter matrix Data
iris_dataframe=pd.DataFrame(Input_train,columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe,c=output_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=.8,cmap=mglearn.cm3)
#plt.show()

## knn is a capsuled algorithm by Train data predicting the new data point
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(Input_train,output_train)  # Making the model by using training dataset,  knn is changed itself after doing method knn.fit

# predicting by inputting random new_data
new=np.array([[1,1,1,1]])
prediction=knn.predict(new)
#print(new.shape)
#print("prediction:{}".format(prediction))
#print("name:{}".format(iris_dataset['target_names'][prediction]))

y_real=knn.predict(Input_train)
y_pred=knn.predict(Input_test)
print("real:{}\n".format(output_test))
print("pred:{}\n".format(y_pred))
print("Test_Set correction degree: {:.2f}".format(np.mean(output_test == y_pred)))

