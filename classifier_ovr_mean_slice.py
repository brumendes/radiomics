import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from itertools import cycle

# Read dataframe
df = pd.read_csv('./results/2D/radiomic_features_2D.csv', skipinitialspace=True, na_values='scalar')

df_mean = df.groupby('CaseNumber').mean()

# Get X, y variables
y = df_mean['Grade']
X = df_mean.drop(['Grade'], axis=1)

# Binarize y for ovr
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Create train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# Build pipeline with StandardScaler, PCA and Ovr classifier and compute score
clf = make_pipeline(
    StandardScaler(),
    PCA(n_components=6),  
    OneVsRestClassifier(svm.SVC(kernel='sigmoid'))
    )
y_score = clf.fit(X_train, y_train).decision_function(X_test)

"""
Obtained Results
Feature     |        AUROC       |
Class       | 0    | 1    | 2    | 
First Order | 0.88 | 0.71 | 0.88 |
Shape       | 0.62 | 0.50 | 0.12 |
GLCM        | 0.62 | 0.57 | 0.75 |
GLRLM       | 0.50 | 0.71 | 0.12 |
GLSZM       | 0.54 | 0.64 | 0.12 | 
GLDM        | 0.50 | 0.64 | 0.00 |
NGTDM       | 0.38 | 0.14 | 0.25 |
ALL         | 0.38 | 0.57 | 0.12 |
For volumetric data we get only 3 images of class 0. 
This number is very low to apply any synthetic data generation algorithm as SMOTE accurately.
"""

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

lw = 2

## ROC curves for the multilabel problem
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]),color='deeppink', linestyle=':', linewidth=4)
plt.plot(fpr["macro"], tpr["macro"],label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Volumetric Features')
plt.legend(loc="lower right")
plt.show()