import cv2
import torch
import pickle
import pydicom
import numpy as np

def load_dicom(file_path):
    dicom = pydicom.dcmread(file_path)
    pixel_array = dicom.pixel_array.astype(np.float32)
    return pixel_array


class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, df, subset: str, modality: str, transform=None, radiomics_path=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.subset = subset
        self.modality = modality

        if 'radiomics' in modality and radiomics_path:
            with open(radiomics_path, 'rb') as f:
                self.radiomics_dict = pickle.load(f)
        else:
            self.radiomics_dict = {}

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row['ID']
        img = load_dicom(row['image_path'])

        mask = np.zeros_like(img, dtype=np.uint8)
        if row['points'] != "None":
            for lesion in row['points']:
                contour = np.array([[p['x'], p['y']] for p in lesion], dtype=np.int32)
                cv2.fillPoly(mask, [contour], 1)

        img = img.astype(np.uint8)
        if self.transform and self.subset == 'train':
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        img = img.astype(np.float32) / 255.0

        if 'radiomics' in self.modality:
            radiomic_feats = self.radiomics_dict.get(patient_id, np.zeros((1,)))
        else:
            radiomic_feats = np.zeros((1,))

        tabular_feats = row.iloc[7:].values.astype(np.float32)
        target = row['subtype']

        return {
            'patient_id': patient_id,
            'image': torch.stack([torch.tensor(img)] * 3, dim=0),  # 3 x H x W
            'mask': torch.tensor(mask, dtype=torch.long),
            'clinical': torch.tensor(tabular_feats[:7], dtype=torch.float32),
            'radiomics': torch.tensor(radiomic_feats, dtype=torch.float32),
            'metalesion': torch.tensor(tabular_feats[7:], dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.long),
        }

    def __len__(self):
        return len(self.df)

