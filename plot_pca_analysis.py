from os import name
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt

# Read dataframe
df = pd.read_csv('./results/pca_analysis_2D_smote.csv', skipinitialspace=True, na_values='scalar')

linear  = df.loc[df['kernel'] == 'linear']
poly = df.loc[df['kernel'] == 'poly']
rbf = df.loc[df['kernel'] == 'rbf']
sigmoid = df.loc[df['kernel'] == 'sigmoid']

max_idx = df['macro'].idxmax()

print(df.loc[max_idx, 'kernel'])
print(max_idx)
print(df.loc[max_idx, '0'])
print(df.loc[max_idx, '1'])
print(df.loc[max_idx, '2'])

test = df.loc[max_idx, 'macro']

plt.figure()
plt.plot(linear['n_components'], linear['macro'], linestyle='dashed', label='linear')
plt.plot(poly['n_components'], poly['macro'], linestyle='dashed', label='poly')
plt.plot(rbf['n_components'], rbf['macro'], linestyle='dashed', label='rbf')
plt.plot(sigmoid['n_components'], sigmoid['macro'], linestyle='dashed', label='sigmoid')
plt.xlabel("Number of components")
plt.ylabel("AUROC")
plt.legend(loc="lower right")
plt.annotate(str(round(test, 2)), (linear.loc[max_idx, 'n_components'], linear.loc[max_idx, 'macro']))
plt.tight_layout()
plt.show()