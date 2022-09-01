import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('./results/3D/radiomic_features_3D.csv', skipinitialspace=True, na_values='scalar')

# Compute the correlation matrix using Pearson Correlation Coefficient
corr = df.corr().abs()

# Get correlation with Grade and filter for features higher then 0.25
corr_target = corr["Grade"]
relevant_features = list(corr_target[corr_target>0.25].index)
new_corr = corr.loc[relevant_features, relevant_features]

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(new_corr, dtype=bool))

"""
Relevant Features Correlation
.............................................
original_shape_Sphericity            0.414619
original_shape_SurfaceVolumeRatio    0.338233
original_firstorder_Kurtosis         0.315297
original_firstorder_Skewness         0.274911
original_glcm_Correlation            0.333388
Grade                                1.000000 
.............................................
Sphericity is highly correlated with SurfaceVolumeRatio: -0.660728
Kurtosis is highly correlated with Skewness: 0.92063

Ideal feature set: Sphericity, Kurtosis, GLCMCorrelation
"""

labels = []

for x in relevant_features:
    sep = x.split('_')
    if len(sep)>1:
        labels.append(sep[2])
    else:
        labels.append(sep[0])

# Correlation Matrix
heat = sns.heatmap(new_corr, annot=True, mask=mask, cmap='coolwarm', square=True, linewidths=.5, xticklabels=labels, yticklabels=labels, cbar=False)
plt.xticks(rotation=20)

# Rename relevant features to PairPlot
new_names = {}
for idx, x in enumerate(relevant_features):
    new_names[x] = labels[idx]

df_renamed = df.rename(columns=new_names, inplace = False)

# PairPlot
# pair = sns.pairplot(
#     df_renamed[labels], 
#     vars=df_renamed[labels].columns[:-1], 
#     hue='Grade', 
#     diag_kind="kde",
#     palette='coolwarm'
#     )
# pair.map_lower(sns.kdeplot, levels=3)
plt.show()