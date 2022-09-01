import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('./results/3D/radiomic_features_3D.csv', skipinitialspace=True, na_values='scalar')

# # Drop CaseNumber and Grade from dataset
# df = df.drop(['CaseNumber', 'Grade'], axis=1)

# Normalize dataset
sc = StandardScaler()
df_transform = sc.fit_transform(df)
df_norm = pd.DataFrame(df_transform, columns=df.columns)

# Principal Component analysis
pca = PCA(n_components=3)
pca.fit_transform(df_norm)
df_pca = pd.DataFrame(data=pca.fit_transform(df_norm), columns = ['PC1', 'PC2', 'PC3'])
"""
Results:
Explained variation per principal component: [0.39672967 0.17230449 0.13936471]
0.29160113 is lost in dimensionality reduction
"""

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
targets = [0, 1, 2]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = df['Grade'] == target
    ax.scatter(
        xs=df_pca.loc[indicesToKeep, 'PC1'],
        ys=df_pca.loc[indicesToKeep, 'PC2'],
        zs=df_pca.loc[indicesToKeep, 'PC3'],
        c = color
    )
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.legend(targets)
fig.tight_layout()

plt.show()