import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor, logger, logging
import six
import matplotlib.pyplot as plt

paramsFile = 'settings3D.yaml'

vol_dir = r'C:\Users\Mendes\Projectos\PythonProjects\Radiomics\volumes'

output_dir = r'C:\Users\Mendes\Projectos\PythonProjects\Radiomics\results\featureMaps\2'

testCase = 'CTV-2'

logger.setLevel(logging.DEBUG)

handler = logging.FileHandler(filename='./results/featureMaps/feat_maps_log.txt', mode='w')
formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

ct_path = os.path.join(vol_dir, '1.nrrd')
struct_path = os.path.join(vol_dir, '1-CTV1-label.nrrd')

# Feature maps extraction for highest correlation features
"""
firstorder=['Kurtosis', 'Skewness'] --> [0.315297, 0.274911]
glcm=['Correlation'] --> [0.333388]
shape=['Sphericity', 'SurfaceVolumeRatio'] --> [0.414619, 0.338233] --> pyradiomics does not compute shape feature maps

Correlation: Correlation is a value between 0 (uncorrelated) and 1 (perfectly correlated) showing the linear dependency of gray level values to their respective voxels in the GLCM

Kurtosis: Kurtosis is a measure of the ‘peakedness’ of the distribution of values in the image ROI. A higher kurtosis implies that the mass of the distribution is concentrated towards the tail(s) rather than towards the mean. A lower kurtosis implies the reverse: that the mass of the distribution is concentrated towards a spike near the Mean value

Skewness: Skewness measures the asymmetry of the distribution of values about the Mean value. Depending on where the tail is elongated and the mass of the distribution is concentrated, this value can be positive or negative.
"""
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.loadParams(paramsFile)
extractor.disableAllFeatures()
extractor.enableFeaturesByName(firstorder=['Kurtosis', 'Skewness'], glcm=['Correlation'])
features = extractor.execute(ct_path, struct_path, voxelBased=True)

for featureName, featureValue in six.iteritems(features):
  if isinstance(featureValue, sitk.Image):
    sitk.WriteImage(featureValue, os.path.join(output_dir, '%s_%s.nrrd' % (testCase, featureName)))
    print('Computed %s, stored as "%s_%s.nrrd"' % (featureName, testCase, featureName))
  else:
    print('%s: %s' % (featureName, featureValue))