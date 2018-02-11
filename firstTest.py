import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import xlrd
from IPython.display import clear_output, Image, display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pandas import ExcelWriter
from pandas import ExcelFile
from openpyxl import load_workbook
from numpy import genfromtxt

dataset_file_name = '/home/jungjunkim/Python_Library_Machine_Learning/Dataset/dataset.csv'
target_file_name = '/home/jungjunkim/Python_Library_Machine_Learning/Dataset/targetset.csv'

dataset=np.genfromtxt(dataset_file_name,delimiter=',')
target=np.genfromtxt(target_file_name,delimiter=',')



print(format(dataset.shape))
print(format(target.shape))




