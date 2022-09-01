import pandas as pd
import matplotlib.pyplot as plt

# Read dataframe
df = pd.read_csv('./results/3D/radiomic_features_3D.csv', skipinitialspace=True, na_values='scalar')

# Get X, y variables
y = df['Grade']
X = df.drop(['CaseNumber', 'Grade'], axis=1)

X = X[[
    'original_shape_Sphericity', 
    'original_firstorder_Kurtosis',
    'original_glcm_Correlation'
    ]]

plt.scatter(X['original_firstorder_Kurtosis'], X['original_glcm_Correlation'])
plt.scatter(X['original_firstorder_Kurtosis'], X['original_shape_Sphericity'])
# plt.scatter(X['original_glcm_Correlation'], y)

plt.show()