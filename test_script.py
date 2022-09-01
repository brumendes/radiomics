import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn import svm
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from itertools import cycle

df = pd.read_csv('./results/2D/radiomic_features_2D.csv', skipinitialspace=True, na_values='scalar', index_col=0)

# Stratify datasets in order to assure evenly distributed samples
df_0 = df[df['Grade'] == 0]
df_1 = df[df['Grade'] == 1]
df_2 = df[df['Grade'] == 2]

y_0 = df_0['Grade']
y_1 = df_1['Grade']
y_2 = df_2['Grade']

X_0 = df_0.drop(['Grade'], axis=1)
X_1 = df_1.drop(['Grade'], axis=1)
X_2 = df_2.drop(['Grade'], axis=1)

cases_0 = X_0['CaseNumber']
cases_1 = X_1['CaseNumber']
cases_2 = X_2['CaseNumber']

train_index_0, test_index_0 = next(GroupShuffleSplit(n_splits=2, train_size=.5, random_state=0).split(X_0, y_0, cases_0))
train_index_1, test_index_1 = next(GroupShuffleSplit(n_splits=2, train_size=.5, random_state=0).split(X_1, y_1, cases_1))
train_index_2, test_index_2 = next(GroupShuffleSplit(n_splits=2, train_size=.5, random_state=0).split(X_2, y_2, cases_2))

X_train_0, X_test_0 = X_0.iloc[train_index_0], X_0.iloc[test_index_0]
X_train_1, X_test_1 = X_1.iloc[train_index_1], X_1.iloc[test_index_1]
X_train_2, X_test_2 = X_2.iloc[train_index_2], X_2.iloc[test_index_2]

y_train_0, y_test_0 = y_0[train_index_0], y_0[test_index_0]
y_train_1, y_test_1 = y_1[train_index_1], y_1[test_index_1]
y_train_2, y_test_2 = y_2[train_index_2], y_2[test_index_2]

X_train = pd.concat([X_train_0, X_train_1, X_train_2])
X_test = pd.concat([X_test_0, X_test_1, X_test_2])

y_train = pd.concat([y_train_0, y_train_1, y_train_2])
y_test = pd.concat([y_test_0, y_test_1, y_test_2])