import os
import numpy as np
import pandas as pd
import radiomics
from radiomics import featureextractor, logger, logging
import SimpleITK as sitk

params = 'settings2D.yaml'
vol_dir = r'C:\Users\Mendes\Projectos\PythonProjects\Radiomics\volumes'
input_csv = 'test_cases.csv'
output_file_path = './results/2D/radiomic_features_2D.csv'
progress_filename = './results/2D/rad_feat_2D_log.txt'

logger.setLevel(logging.INFO)

handler = logging.FileHandler(filename=progress_filename, mode='w')
formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

radiomics.setVerbosity(logging.CRITICAL)
logger.info('pyradiomics version: %s', radiomics.__version__)
logger.info('Loading CSV...')

try:
    cases_list = pd.read_csv(input_csv)
    n_cases = len(cases_list)
except Exception:
    logger.error('CSV READ FAILED', exc_info=True)
    exit(-1)

logger.critical('Radiomic features extraction from slices.')
logger.critical('Number of Cases: %d', n_cases)

results = []

for index, case in cases_list.iterrows():
    case_number = case['CaseNumber']
    grade = case['Grade']
    logger.critical('Processing Case %s of %s', case_number, n_cases)

    ct_volume_file_path = os.path.join(vol_dir, case['Image'])
    mask_volume_file_path = os.path.join(vol_dir, case['Mask'])

    ct_volume = sitk.ReadImage(ct_volume_file_path)
    mask_volume = sitk.ReadImage(mask_volume_file_path)

    extractor = featureextractor.RadiomicsFeatureExtractor(params)

    for idx in range(1, mask_volume.GetSize()[2]):
        sitk_img = sitk.JoinSeries(ct_volume[:,:,idx])
        sitk_mask = sitk.JoinSeries(mask_volume[:,:,idx])
        try:
            result = extractor.execute(sitk_img, sitk_mask)
            result['CaseNumber'] = case_number
            result['Grade'] = grade
            results.append(result)
        except Exception:
            pass

df_features = pd.DataFrame(data=[f for f in results])

df_features.to_csv(output_file_path)

logger.critical('All slices processed. Results saved.')