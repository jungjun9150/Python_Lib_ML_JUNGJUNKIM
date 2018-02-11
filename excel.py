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

file_name = '/home/jungjunkim/Python_Library_Machine_Learning/dataset.csv'
csv = pd.read_csv(file_name)
print(csv.values.shape)
