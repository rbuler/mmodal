import cv2
import torch
import random
import pydicom
import numpy as np
import pandas as pd
from RadiomicExtractor import RadiomicsExtractor


def load_dicom(file_path):
    dicom = pydicom.dcmread(file_path)
    pixel_array = dicom.pixel_array.astype(np.float32)
    return torch.tensor(pixel_array, dtype=torch.float32)


class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, df, subset: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.subset = subset
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        patient_id = row['ID']
        img = load_dicom(row['image_path'])
        
        
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        flag = True
        if row['points'] == "None":
            flag = False
        else:
            points = row['points']
            contour = np.array([[p['x'], p['y']] for p in points], dtype=np.int32)
            cv2.fillPoly(mask, [contour], 1)

        img = img.numpy()
        if self.transform and self.subset == 'train':
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        # TODO 
        # add radiomics features extraction here
        if flag:
            extractor = RadiomicsExtractor(param_file='params.yml')
            list_of_dicts = [{'image': img, 'mask': mask}]
            features = extractor.serial_extraction(list_of_dicts)
            features = pd.DataFrame(features)
            radiomic_imputed = 0
        else:
            features = pd.DataFrame(np.zeros((1, 102)))
            radiomic_imputed = random.choice([0, 1])

        tabular_feats = row.iloc[6:].values.astype(np.float32)
        radiomic_feats = features.values.astype(np.float32)
        radiomic_feats = np.squeeze(radiomic_feats)

        # TODO
        # apply dropout to tabular features only during training
        # maybe add fixed ids to be dropped out ?????
        # think of the best way to do this
        dropout = 0.33
        if np.any(tabular_feats[7:] == 1) and self.subset == 'train':
            if random.random() < dropout:
                tabular_feats[7:] = 0
                radiomic_feats[:] = 0
                radiomic_imputed = 1


        img = img.astype(np.float32)
        img = img / 255.0  # Normalize to [0, 1]
        target = row['classification'].astype(np.float32)

        return {
            'patient_id': patient_id,
            'image': torch.stack([torch.tensor(img, dtype=torch.float32)] * 3, dim=0),  # Convert to 3 x H x W
            'mask': torch.tensor(mask, dtype=torch.long),
            'clinical': torch.tensor(tabular_feats[:7], dtype=torch.float32),
            'radiomics': torch.tensor(radiomic_feats, dtype=torch.float32),
            'metalesion': torch.tensor(tabular_feats[7:], dtype=torch.float32),
            'radiomics_imputed': torch.tensor(radiomic_imputed, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
        }

    def __len__(self):
        return len(self.df)
