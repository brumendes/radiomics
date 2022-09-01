import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from tools import Classifier
from sklearn.decomposition import PCA, SparsePCA, KernelPCA, TruncatedSVD, IncrementalPCA, MiniBatchSparsePCA, FactorAnalysis
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import FastICA

# Read features and gleason scores files to dataframe
df = pd.read_csv('./results/TextureSession.csv')

df_gleason = pd.read_csv('./info/patients_scores.csv')

merged_df = pd.merge(df, df_gleason, on='INFO_PatientID', how='inner')

######################################## Correlation Study ###########################################
# import seaborn as sns

# merged_df['RiskGroup'].replace({
#     'Low/VeryLow': 0, 
#     'Intermediate(Favorable/Unfavorable)': 1,
#     'High/VeryHigh': 2,
#     }, inplace=True)

# corr = merged_df.corr().abs()
# corr_target = corr['RiskGroup']
# relevant_features = list(corr_target[corr_target>0.25].index)
# new_corr = corr.loc[relevant_features, relevant_features]

# # Generate a mask for the upper triangle
# mask = np.triu(np.ones_like(new_corr, dtype=bool))

# labels = []

# for x in relevant_features:
#     sep = x.split('_')
#     if len(sep)>1:
#         labels.append(sep[1])
#     else:
#         labels.append(sep[0])

# heat = sns.heatmap(new_corr, annot=True, mask=mask, cmap='coolwarm', square=True, linewidths=.5, xticklabels=labels, yticklabels=labels, cbar=False)
# plt.xticks(rotation=20)

# plt.show()

############################################################################################################

################# If correlation is poor, build a classifier with PCA and OvR approach #####################
lb = LabelBinarizer()
y = lb.fit_transform(merged_df['RiskGroup'])

X = merged_df.loc[:,merged_df.columns.str.startswith((
    'CONVENTIONAL_HU', 
    'DISCRETIZED_HU', 
    'DISCRETIZED_HIST', 
    'SHAPE',
    'GLCM',
    'GLRLM',
    'NGLDM',
    'GLZLM',
    ))]

n_samples = len(X.index)
n_features = len(X.columns)

max_components = int(min(n_samples*0.8, n_features))

n_jobs = 30

pca_methods = [PCA, SparsePCA, KernelPCA, TruncatedSVD, IncrementalPCA]

clf = Classifier(X, y)

rows = []

# Evaluate optimal number of components for maximum ROC
for pca in pca_methods:
    for n in range(1, max_components):
        clf.run(n, n_jobs, pca, stratified=False)
        roc_values = clf.get_roc_score()
        rows.append([pca, n, roc_values[0], roc_values[1], roc_values[2]])
        print('Number of components:{}'.format(n) + '  |  ' + 'ROC:{}'.format(clf.get_roc_score()))

df = pd.DataFrame(rows, columns=["PCA Method", "Components", "Low/VeryLow", "Intermediate", "High/VeryHigh"])

df.to_csv('./results/MDPI_analysis_lifex.csv', index=False)
