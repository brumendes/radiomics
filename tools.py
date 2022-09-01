import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, auc, roc_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from itertools import cycle
from imblearn.over_sampling import SMOTE

class Classifier:
    """
    This Model builds a pipeline with a StandardScaler, PCA and SVC in a OneVsRest approach.
    It prints the Area under the ROC for each class and precision, recall and f1-score.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.roc_score = None
        self.ap_score = None

    def run(self, n_components, n_jobs, pca_method, stratified=False, synthetic=False):
        if stratified:
            skf = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=0)

            for train_index, test_index in skf.split(self.X, self.y):
                X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=.2, random_state=0)
        if synthetic:
            # Balance dataset with SMOTE
            X_train, y_train = SMOTE().fit_resample(X_train, y_train)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('selection', pca_method(n_components=n_components)),
            ('classifier', OneVsRestClassifier(SVC(kernel='linear'), n_jobs=n_jobs)),
            # ('classifier', OneVsRestClassifier(LogisticRegression(C=C, penalty='l1', solver='liblinear', max_iter=10000)))
        ])
        y_score = pipe.fit(X_train, y_train).decision_function(X_test)
        target_names = ['Low/VeryLow', 'Intermediate', 'High/VeryHigh']
        y_pred = pipe.predict(X_test)
        self.ap_score = average_precision_score(y_test, y_score, average='weighted')
        self.roc_score = roc_auc_score(y_test, y_score, average=None, multi_class='ovr', labels=target_names)

    def get_roc_score(self):
        return self.roc_score

    def get_ap_score(self):
        return self.ap_score


class MulticlassROCDisplay:
    def __init__(self, y_test, y_score, n_classes):
        self.y_test = y_test
        self.y_score = y_score
        self.n_classes = n_classes

    def run(self):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], self.y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(self.y_test.ravel(), self.y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        lw = 2

        ## ROC curves for the multilabel problem
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= self.n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]),color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"],label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

        for i, color in zip(range(self.n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Volumetric Features')
        plt.legend(loc="lower right")
        plt.show()