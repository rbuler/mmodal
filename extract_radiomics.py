# %%
import cv2
import pickle
import pydicom
import logging
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from RadiomicExtractor import RadiomicsExtractor
from sklearn.feature_selection import VarianceThreshold


warnings.filterwarnings('ignore')


logger_radiomics = logging.getLogger("radiomics")
logger_radiomics.setLevel(logging.ERROR)

def load_dicom(file_path):
    dicom = pydicom.dcmread(file_path)
    return dicom.pixel_array.astype(np.float32)


def main(df, output_path='radiomics.pkl', param_file='params.yml'):
    extractor = RadiomicsExtractor(param_file=param_file)
    all_features = {}
    all_concat_features = []
    all_targets = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
       
        patient_id = row['ID']
        img = load_dicom(row['image_path'])
        points = row['points']
        radiomic_feats_list = []

        if points == "None":
            all_features[patient_id] = None
            continue

        for lesion in points:
            lesion_mask = np.zeros_like(img, dtype=np.uint8)
            contour = np.array([[p['x'], p['y']] for p in lesion], dtype=np.int32)
            cv2.fillPoly(lesion_mask, [contour], 1)

            features = extractor.serial_extraction([{'image': img, 'mask': lesion_mask}])
            df_feats = pd.DataFrame(features)
            feat_array = df_feats.values.astype(np.float32)

            if feat_array.size > 0:
                radiomic_feats_list.append(feat_array)
                all_concat_features.append(feat_array)

        if radiomic_feats_list:
            all_features[patient_id] = np.concatenate(radiomic_feats_list, axis=0)
        else:
            all_features[patient_id] = None

    global_feats = np.concatenate([f for f in all_concat_features if f is not None], axis=0)
    all_targets = np.array(all_targets)

    selector = VarianceThreshold(threshold=0.0)
    global_feats_reduced = selector.fit_transform(global_feats)
    print(f"Features reduced: {global_feats.shape[1]} → {global_feats_reduced.shape[1]}")

    scaler = StandardScaler()
    _ = scaler.fit_transform(global_feats_reduced)

    for patient_id, feats in all_features.items():
        if feats is not None:
            reduced = selector.transform(feats)
            scaled = scaler.transform(reduced)
            all_features[patient_id] = scaled
        else:
            all_features[patient_id] = None

    with open(output_path, 'wb') as f:
        pickle.dump(all_features, f)
    print(f"Radiomics saved to: {output_path}")


if __name__ == "__main__":
    import os
    import yaml
    import typing
    import argparse

    def get_args_parser(path: typing.Union[str, bytes, os.PathLike]):
        help = '''path to .yml config file
        specyfying datasets/training params'''

        parser = argparse.ArgumentParser()
        parser.add_argument("--config_path", type=str,
                            default=path,
                            help=help)
        return parser


    parser = get_args_parser('config.yml')
    args, unknown = parser.parse_known_args()
    with open(args.config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)


    df = pd.read_pickle(config['dir']['df'])
    main(df, output_path=config['dir']['radiomics'])
# %%