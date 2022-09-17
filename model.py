import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('UCI_Credit_Card.csv')
df.head()

rows = len(df.axes[0])
cols = len(df.axes[1])
print("Number of Rows:- ", rows)
print("Number of Cols:-", cols)

print(df.info())

df.SEX.value_counts()
df.EDUCATION.value_counts()
df['EDUCATION'].replace([0, 6], 5, inplace = True)
df.EDUCATION.value_counts()

df.MARRIAGE.value_counts()
df['MARRIAGE'].replace(0, 3, inplace = True)
df.MARRIAGE.value_counts()

df.PAY_2.value_counts()
df.PAY_0.value_counts()

df.drop('ID',axis=1, inplace=True)
print(df.columns)

X = df.iloc[:, :-1].values
print(X)

y = df.iloc[:, -1].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

