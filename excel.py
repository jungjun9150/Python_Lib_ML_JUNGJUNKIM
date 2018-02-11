from openpyxl import load_workbook
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import xlrd
from IPython.display import clear_output, Image, display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


excel_file = '/home/jungjunkim/Python_Library_Machine_Learning/dataset.xlsx'
dataset=pd.read_excel(excel_file)
dataset.head()
