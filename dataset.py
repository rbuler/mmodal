import cv2
import torch
import pydicom
import numpy as np
import pandas as pd
from RadiomicExtractor import RadiomicsExtractor


def load_dicom(file_path):
    dicom = pydicom.dcmread(file_path)
    pixel_array = dicom.pixel_array.astype(np.float32)
    return torch.tensor(pixel_array, dtype=torch.float32)


class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, df, subset: str, modality: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.subset = subset
        self.modality = modality
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        patient_id = row['ID']
        img = load_dicom(row['image_path'])
        
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        radiomic_feats_list = []
        points = row['points']
  
        for lesion in points:  # iterate over each lesion
            lesion_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            contour = np.array([[p['x'], p['y']] for p in lesion], dtype=np.int32)
            cv2.fillPoly(lesion_mask, [contour], 1)
            
            has_points = row['points'] != "None"
            is_radiomics = 'radiomics' in self.modality
            if has_points and is_radiomics:
                extractor = RadiomicsExtractor(param_file='params.yml')
                list_of_dicts = [{'image': img.numpy(), 'mask': lesion_mask}]
                features = extractor.serial_extraction(list_of_dicts)
                features = pd.DataFrame(features)
                radiomic_feats_list.append(features.values.astype(np.float32))
            
            mask = cv2.bitwise_or(mask, lesion_mask)

        img = img.numpy().astype(np.uint8)
        if self.transform and self.subset == 'train':
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        if radiomic_feats_list:
            radiomic_feats = np.concatenate(radiomic_feats_list, axis=0)
        else:
            radiomic_feats = np.zeros((1,))

        tabular_feats = row.iloc[7:].values.astype(np.float32)
        # radiomic_feats = np.squeeze(radiomic_feats)
        img = img.astype(np.float32)
        img = img / 255.0  # Normalize to [0, 1]
        target = row['subtype'].astype(np.int64)

        return {
            'patient_id': patient_id,
            'image': torch.stack([torch.tensor(img, dtype=torch.float32)] * 3, dim=0),  # Convert to 3 x H x W
            'mask': torch.tensor(mask, dtype=torch.long),
            'clinical': torch.tensor(tabular_feats[:7], dtype=torch.float32),
            'radiomics': torch.tensor(radiomic_feats, dtype=torch.float32),
            'metalesion': torch.tensor(tabular_feats[7:], dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.long),
        }

    def __len__(self):
        return len(self.df)
