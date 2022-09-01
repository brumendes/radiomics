import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline

# Read dataframe
df = pd.read_csv('./results/2D/radiomic_features_2D.csv', skipinitialspace=True, na_values='scalar', index_col=0)

output_file_path = './results/pca_analysis_2D.csv'

# Get X, y variables
y = df['Grade']
X = df.drop(['Grade'], axis=1)

########### This section is for 2D only ##########
# Group Shuffle: to ensure that the same case is not represented in both testing and training sets - 2D only
cases = X['CaseNumber']
cases_group_shuffle = GroupShuffleSplit(n_splits=2, test_size=.2, random_state=0)

# Create train and test datasets with cases_group_shuffle - 2D only
for train_index, test_index in cases_group_shuffle.split(X, y, cases):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

# Balance dataset with SMOTE - 2D only
# X_train, y_train = SMOTE().fit_resample(X_train, y_train)

# Binarize y for ovr
y_train = label_binarize(y_train, classes=[0, 1, 2])
y_test = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_train.shape[1]

# Remove CaseNumber from X
X_train = X_train.drop(['CaseNumber'], axis=1)
X_test = X_test.drop(['CaseNumber'], axis=1)

########## This is for 3D ##########
# # Create train and test datasets - 3D only
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)


kernels = ['linear', 'poly', 'rbf', 'sigmoid']

results = []

for kernel in kernels:
    for n_cmp in range(1, 36):
        # Build pipeline with StandardScaler, PCA and Ovr classifier and compute score
        clf = make_pipeline(
            StandardScaler(),
            PCA(n_components=n_cmp),  
            OneVsRestClassifier(svm.SVC(kernel=kernel))
            )
        y_score = clf.fit(X_train, y_train).decision_function(X_test)

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
        roc_auc['n_components'] = n_cmp
        roc_auc['kernel'] = kernel
        results.append(roc_auc)


# df_features = pd.DataFrame(data=[f for f in results])

# df_features.to_csv(output_file_path)