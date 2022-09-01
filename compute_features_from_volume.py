import os
import pandas as pd
import radiomics
from radiomics import featureextractor, logger, logging

params = 'settings3D.yaml'
vol_dir = r'C:\Users\Mendes\Projectos\PythonProjects\Radiomics\volumes'
input_csv = 'test_cases.csv'
output_file_path = './results/3D/radiomic_features_3D.csv'
progress_filename = './results/3D/rad_feat_3D_log.txt'

logger.setLevel(logging.INFO)

handler = logging.FileHandler(filename=progress_filename, mode='w')
formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

radiomics.setVerbosity(logging.CRITICAL)
logger.info('pyradiomics version: %s', radiomics.__version__)
logger.info('Loading CSV')

try:
    cases_list = pd.read_csv(input_csv)
    n_cases = len(cases_list)
except Exception:
    logger.error('CSV READ FAILED', exc_info=True)
    exit(-1)

logger.critical('Radiomic features extraction from volume.')
logger.critical('Patients: %d', n_cases)

extractor = featureextractor.RadiomicsFeatureExtractor(params)

results = []

for index, case in cases_list.iterrows():
    case_number = case['CaseNumber']
    grade = case['Grade']
    logger.critical('Processing Case %s of %s', case_number, n_cases)

    ct_volume_file_path = os.path.join(vol_dir, case['Image'])
    mask_volume_file_path = os.path.join(vol_dir, case['Mask'])

    try:
        result = extractor.execute(ct_volume_file_path, mask_volume_file_path)
        result['CaseNumber'] = case_number
        result['Grade'] = grade
        results.append(result)
    except Exception:
        pass

df_features = pd.DataFrame(data=[f for f in results])

df_features.to_csv(output_file_path)

logger.critical('All volumes processed. Results saved.')



