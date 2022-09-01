import os
from pydicom import dcmread
import numpy as np
import pandas as pd
from matplotlib.path import Path
from radiomics import featureextractor, firstorder
import SimpleITK as sitk
import matplotlib.pyplot as plt

test_cases = pd.read_csv('test_cases.csv', skipinitialspace=True)

params = 'settings2D.yaml'

data_folder = r"C:\Users\Mendes\Pictures\RadiomicsDataSet"

output_folder = './results/2D/all_features.csv'

results = []

for study_idx, study in enumerate(os.listdir(data_folder)):

    study_folder = os.path.join(data_folder, study)

    ct_files = []
    ct_files_path_list = []

    extractor = featureextractor.RadiomicsFeatureExtractor(params)

    for file_path in os.listdir(study_folder):
        dcm = dcmread(os.path.join(study_folder, file_path))
        if dcm.Modality == 'CT':
            ct_files.append(dcm)
            ct_files_path_list.append(os.path.join(study_folder, file_path))
        elif dcm.Modality == 'RTSTRUCT':
            struct_file = dcm
        else:
            pass

    print('CaseNumber: %s' % str(study_idx + 1))
    print('Available volumes in series.')
    for struct_set in struct_file.StructureSetROISequence:
        print('%s:%s' % (struct_set.ROINumber, struct_set.ROIName))

    selected_volume = input("Please select a volume number to perform feature extraction:")

    for roi in struct_file.ROIContourSequence:
        if roi.ReferencedROINumber == int(selected_volume):
            for contour_idx, contour in enumerate(roi.ContourSequence):
                for contour_image in contour.ContourImageSequence:
                    reference_image_uid = contour_image.ReferencedSOPInstanceUID
                for idx, ct_file in enumerate(ct_files):
                    if ct_file.SOPInstanceUID == reference_image_uid:
                        data_spacing = ct_file.PixelSpacing
                        case_number = test_cases.loc[study_idx]['CaseNumber']
                        grade = test_cases.loc[test_cases['CaseNumber'] == case_number, 'Grade'].iloc[0]
                        sitk_img = sitk.GetImageFromArray(ct_file.pixel_array)
                        sitk_img.SetSpacing((data_spacing[0], data_spacing[1], 1))
                        sitk_img = sitk.JoinSeries(sitk_img)
                contour_data = contour.ContourData
                contour_data = np.array([contour_data[i:i + 3] for i in range(0, len(contour_data), 3)])
                path = Path(contour_data[:,[1,0]]*10+256)
                x, y = np.mgrid[:512, :512]
                points = np.hstack((x.reshape(-1, 1), y.reshape(-1,1)))
                mask = path.contains_points(points).reshape(512, 512)
                sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint16))
                sitk_mask.SetSpacing((data_spacing[0], data_spacing[1], 1))
                sitk_mask = sitk.JoinSeries(sitk_mask)
                print('Processing slice %s of %s...' % (str(contour_idx + 1), len(roi.ContourSequence)))
                try:
                    result = extractor.execute(sitk_img, sitk_mask)
                    result['CaseNumber'] = case_number
                    result['Grade'] = grade
                    results.append(result)
                except Exception:
                    print('FEATURE EXTRACTION FAILED!')
                    pass

df_features = pd.DataFrame(data=[f for f in results])

df_features.to_csv(output_folder)