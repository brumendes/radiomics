import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from tools import Classifier
from sklearn.decomposition import PCA

# Read features and gleason scores files to dataframe
df = pd.read_csv('./results/3D/radiomic_features_3D.csv')

# Get X, y variables
X = df.drop(['CaseNumber', 'Grade'], axis=1)

lb = LabelBinarizer()
y = lb.fit_transform(df['Grade'])

n_samples = len(X.index)
n_features = len(X.columns)

max_components = int(min(n_samples*0.8, n_features))

n_jobs = 1

pca_methods = [PCA]
pca_names = ['PCA']

clf = Classifier(X, y)

rows = []

# Evaluate optimal number of components for maximum ROC
for idx, pca in enumerate(pca_methods):
    pca_str = pca_names[idx]
    for n in range(1, max_components):
        clf.run(n, n_jobs, pca, stratified=False)
        roc_values = clf.get_roc_score()
        ap_values = clf.get_ap_score()
        rows.append(["PyRadiomics", pca_str, n, roc_values[0], roc_values[1], roc_values[2], ap_values])
        print('Number of components:{}'.format(n) + '  |  ' + 'ROC:{}'.format(clf.get_roc_score()))

df = pd.DataFrame(rows, columns=["Features", "PCA Method", "Components", "Low/VeryLow", "Intermediate", "High/VeryHigh", 'Average Precision'])

df.to_csv('./results/RecPad21_analysis_3D.csv', index=False)
