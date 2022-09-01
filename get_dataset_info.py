import os
from pydicom import dcmread
import pandas as pd

data_folder = r"C:\Users\Mendes\Pictures\RadiomicsDataSet"

file_list = []

data = {
    'StudyDate': [],
    'BirthDate': [],
    'RTPlanLabel': [], 
    'RTPlanDate': [], 
    'PlanIntent': []
    }

ct_data = {
    'PatientID': [],
    'Manufacturer': [],
    'CTScan': [],
    'SliceThickness': [],
    'Kvp': []
}

for root, directories, filenames in os.walk(data_folder):
    for filename in filenames:
        file_path = os.path.join(root, filename)
        file_list.append(file_path)

for idx, file_path in enumerate(file_list):
    dataset = dcmread(file_path, force=True)
    if file_path.endswith('.dcm'):
        if dataset.Modality == 'RTPLAN':
            data['StudyDate'].append(dataset.StudyDate)
            data['BirthDate'].append(dataset.PatientBirthDate)
            data['RTPlanLabel'].append(dataset.RTPlanLabel)
            data['RTPlanDate'].append(dataset.RTPlanDate)
            data['PlanIntent'].append(dataset.PlanIntent)
        if dataset.Modality == 'CT':
            ct_data['PatientID'].append(dataset.PatientID)
            ct_data['Manufacturer'].append(dataset.Manufacturer)
            ct_data['CTScan'].append(dataset.ManufacturerModelName)
            ct_data['SliceThickness'].append(dataset.SliceThickness)
            ct_data['Kvp'].append(dataset.KVP)

df_data = pd.DataFrame(data=data)
df_ct_data = pd.DataFrame(data=ct_data)

min_study_date = str(df_data['StudyDate'].min())[:4]
max_study_date = str(df_data['StudyDate'].max())[:4]

min_birth_date = str(df_data['BirthDate'].min())[:4]
max_birth_date = str(df_data['BirthDate'].max())[:4]

print('Studies from ' + min_study_date + ' to ' + max_study_date)
print('Patients aged from ' + min_birth_date + ' to ' + max_birth_date)

print(df_ct_data['Kvp'].unique())