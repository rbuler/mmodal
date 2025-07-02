 # %% IMPORTS AND SETTINGS
import os
import yaml
import typing
import torch
import pydicom
import logging
import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
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

# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = config['device'] if torch.cuda.is_available() else 'cpu'

def load_data(file_path, sheet_name=1, header_rows=3):
    data = pd.read_excel(file_path, sheet_name=sheet_name, header=list(range(header_rows)))
    data.columns = [col[0] if "Unnamed" in str(col[1]) else f"{col[0]}_{col[1]}" for col in data.columns]
    return data

# %%
csv_path = config['dir']['csv']
csv_original_path = config['dir']['csv_original']
img_path = config['dir']['img']
seg_path = config['dir']['seg']
# %%

df = load_data(csv_path, sheet_name=1, header_rows=2)
df = df[~df['classification'].isin(['Invisible', 'Exclusion'])]
for id in df["ID"].unique():
    id_rows = df[df["ID"] == id]
    if "L" in id_rows["LeftRight"].values and "R" in id_rows["LeftRight"].values:
        age_values = id_rows["Age"].dropna().unique()
        if len(age_values) == 1:
            df.loc[df["ID"] == id, "Age"] = age_values[0]
original_df = pd.read_csv(csv_original_path, header=None)
original_df = original_df[0].str.split(';', expand=True)
original_df = original_df.iloc[:, :-1]
original_df.columns = ['ID', 'LeftRight', 'Age', 'number', 'abnormality', 'classification', 'subtype']

for id in df["ID"].unique():
    if pd.isna(df.loc[df["ID"] == id, "Age"]).all():
        age_from_data = original_df.loc[original_df["ID"] == id, "Age"].dropna().unique()
        if len(age_from_data) == 1:
            df.loc[df["ID"] == id, "Age"] = age_from_data[0]

# %%

seg_df = pd.DataFrame(columns=["ID", "view", "laterality", "points", "label"])

seg_df = seg_df.sort_values(by="ID").reset_index(drop=True)
seg_dict = {}

for file_name in os.listdir(seg_path):
    if file_name.endswith(".json"):
        file_path = os.path.join(seg_path, file_name)
        with open(file_path, 'r') as f:
            json_data = yaml.safe_load(f)
            for lesion in json_data:
                if "cgPoints" in lesion and "label" in lesion:
                    points = lesion["cgPoints"]
                    label = lesion["label"].strip()  # 'calc', 'calc '
                    file_key = (file_name.split("_")[0], file_name.split("_")[1], file_name.split("_")[2].split("_")[0])
                    if file_key not in seg_dict:
                        seg_dict[file_key] = {"points": [], "label": []}
                    seg_dict[file_key]["points"].append(points)
                    seg_dict[file_key]["label"].append(label)

rows = []
for (id, view, laterality), data in seg_dict.items():
    rows.append({
        "ID": id,
        "view": view,
        "LeftRight": laterality,
        "points": data["points"],
        "label": data["label"]
    })

seg_df = pd.concat([seg_df, pd.DataFrame(rows)], ignore_index=True)
seg_df = seg_df.sort_values(by="ID").reset_index(drop=True)



# %% 
# drop patients with both benign and malignant lesions

# simul_bg_mg = ["D1-0397",
#                "D1-0869",
#                "D2-0033",
#                "D2-0116",
#                "D2-0133",
#                "D2-0185",
#                "D2-0637"]

# df = df[~df["ID"].isin(simul_bg_mg)]
# seg_df = seg_df[~seg_df["ID"].isin(simul_bg_mg)]

# %%

df["points"] = None
df["label"] = None

for index, row in df.iterrows():
    matching_rows = seg_df[(seg_df["ID"] == row["ID"]) & (seg_df["LeftRight"] == row["LeftRight"])]
    if not matching_rows.empty:
        df.at[index, "points"] = matching_rows["points"].values[0]
        df.at[index, "label"] = matching_rows["label"].values[0]

# %%

image_df = pd.DataFrame(columns=["ID", "laterality", "view", "image_path"])

for folder_name in os.listdir(img_path):
    folder_path = os.path.join(img_path, folder_name)
    if os.path.isdir(folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                if file_name.endswith(".dcm"):
                    file_path = os.path.join(root, file_name)
                    try:
                        dicom_data = pydicom.dcmread(file_path)
                        laterality = dicom_data.get("ImageLaterality", "Unknown")
                        view = dicom_data.get("ViewPosition", "Unknown")
                        if "ViewCodeSequence" in dicom_data:
                            view = dicom_data.ViewCodeSequence[0].CodeMeaning
                        
                        if view.lower() == "medio-lateral oblique":
                            view = "MLO"
                        elif view.lower() == "cranio-caudal":
                            view = "CC"
                        
                        image_df = pd.concat([image_df, pd.DataFrame({
                            "ID": [folder_name],
                            "laterality": [laterality],
                            "view": [view],
                            "image_path": [file_path]
                        })], ignore_index=True)
                    except Exception as e:
                        logger.error(f"Error reading DICOM file {file_path}: {e}")


# IDs with wrong laterality in DICOM files (checked manually)
wrong_laterality_ids = ['D2-0224', 'D2-0229', 'D2-0642']
for id in wrong_laterality_ids:
    matching_rows = image_df[image_df["ID"] == id]
    if not matching_rows.empty:
        mlo_rows = matching_rows[matching_rows["view"] == "MLO"]
        if len(mlo_rows) == 2:
            left_row = mlo_rows[mlo_rows["laterality"] == "L"]
            right_row = mlo_rows[mlo_rows["laterality"] == "R"]
            if not left_row.empty and not right_row.empty:
                left_path = left_row["image_path"].values[0]
                right_path = right_row["image_path"].values[0]
                image_df.loc[left_row.index, "image_path"] = right_path
                image_df.loc[right_row.index, "image_path"] = left_path

# %%
# filter non malignant cases and add subtype information

df = df[df["classification"].isin(["Malignant"])]
df["subtype"] = None

for index, row in df.iterrows():
    matching_row = original_df[
        (original_df["ID"] == row["ID"]) &
        (original_df["LeftRight"] == row["LeftRight"]) &
        (original_df["classification"] == row["classification"])
    ]
    if not matching_row.empty:
        subtype_value = matching_row["subtype"].dropna().unique()
        if len(subtype_value) == 1:
            df.at[index, "subtype"] = subtype_value[0]
df = df[df["subtype"].apply(lambda x: not isinstance(x, str) or len(x) > 0)]
# %%

if "view" not in df.columns:
    df.insert(2, "view", "MLO")
df["image_path"] = None
for index, row in df.iterrows():
    matching_rows = image_df[(image_df["ID"] == row["ID"]) & (image_df["laterality"] == row["LeftRight"]) & (image_df["view"] == row["view"])]
    if not matching_rows.empty:
        df.at[index, "image_path"] = matching_rows["image_path"].values[0]

df = df.dropna(subset=["image_path"])
df = df.replace({r'\u200b': ''}, regex=True)  # remove zero-width space characters

# %%

columns_to_drop = [
    'classification_Exclusion reasons',
    'Calcification_Clearly Benign Calcifications',
    'Calcification_Other Associated Findings',
    'Other findings_Location',
    'Other findings_Breast Parenchyma',
    'Other findings_Skin',
    'Other findings_Lymph Nodes',
    'Notes',
    'Mass - Additional Lesion_Location',
    'Mass - Additional Lesion_Shape',
    'Mass - Additional Lesion_Margin',
    'Mass - Additional Lesion_Density',
    'Mass - Additional Lesion_Associated calcification',
    'Mass - Additional Lesion_Other Associated Findings',
    'Mass_Associated calcification',                             # too many distinct values, not very specific
    'Mass_Other Associated Findings',
    'Calcification - Additional Lesion_Location',
    'Calcification - Additional Lesion_Morphology',
    'Calcification - Additional Lesion_Distribution',
    'Calcification - Additional Lesion_Clearly Benign Calcifications',
    'BI-RADS\nCategory',
]
df = df.drop(columns=columns_to_drop, errors='ignore')



# %%

cat_cols = [
    'LeftRight', 'Breast density', 
    'Mass_Location', 'Mass_Shape', 'Mass_Margin', 'Mass_Density',
    'Calcification_Location', 'Calcification_Morphology',
    'Calcification_Distribution'
]



# === Canonical label extraction for Mass_Margin ===
def extract_mass_margin_labels(df, col_name="Mass_Margin"):
    canonical_labels = [
        "indistinct",
        "spiculated",
        "microlobulated",
        "amorphous",
        "circumscribed",
        "obscured"
    ]
    
    def clean_text(val):
        if not isinstance(val, str):
            return ""
        return val.lower().strip()
    
    for label in canonical_labels:
        col_flag = f"{col_name}_{label}"
        df[col_flag] = df[col_name].apply(
            lambda x: label in [s.strip() for s in clean_text(x).split("/")]
        )
    
    return df.drop(columns=[col_name])


# === Canonical label extraction for Calcification_Morphology ===
def extract_calcification_morphology_labels(df, col_name="Calcification_Morphology"):
    canonical_labels = [
        "pleomorphic",
        "amorphous",
        "fine and linear",
        "small round"
    ]
    
    def clean_text(val):
        if not isinstance(val, str):
            return ""
        val = val.lower()
        val = val.replace("calcification", "")  # remove word entirely
        return val.strip()
    
    for label in canonical_labels:
        col_flag = f"{col_name}_{label.replace(' ', '_')}"
        df[col_flag] = df[col_name].apply(
            lambda x: label in clean_text(x)
        )
    
    return df.drop(columns=[col_name])


def extract_mass_density_labels(df, col_name="Mass_Density"):
    canonical_labels = ["low density", "equal density", "high density"]
    
    def clean_text(val):
        if not isinstance(val, str):
            return ""
        return val.lower().strip()
    
    for label in canonical_labels:
        col_flag = f"{col_name}_{label.replace(' ', '_')}"
        df[col_flag] = df[col_name].apply(
            lambda x: label in [s.strip() for s in clean_text(x).split("/")]
        )
    
    return df.drop(columns=[col_name])

df = extract_mass_density_labels(df, "Mass_Density")
df = extract_mass_margin_labels(df, "Mass_Margin")
df = extract_calcification_morphology_labels(df, "Calcification_Morphology")
cat_cols = [c for c in cat_cols if c not in ["Mass_Margin", "Mass_Density", "Calcification_Morphology"]]
df_encoded = pd.get_dummies(df, columns=cat_cols)

# %%
scaler = StandardScaler()
df_encoded["Age"] = scaler.fit_transform(df_encoded[["Age"]])
df_encoded["classification"] = df["classification"].replace({"Normal": 0, "Benign": 1, "Malignant": 2})
df_encoded["subtype"] = df["subtype"].replace({
    "Luminal A": 0,
    "Luminal B": 1,
    "HER2-enriched": 2,
    "triple negative": 3
})
non_predictive_columns = ['ID', 'view', 'classification', 'subtype', 'points', 'label', 'image_path']
predictive_columns_order = [
    'Age', 'LeftRight_L', 'LeftRight_R',
    
    'Breast density_extremely dense', 'Breast density_fatty',
    'Breast density_heterogeneous dense', 'Breast density_scattered', 
    
    'Mass_Location_L', 'Mass_Location_M', 'Mass_Location_M-L', 'Mass_Location_S',
    'Mass_Location_U', 'Mass_Location_U-M', 'Mass_Location_Whole area',
    
    'Mass_Shape_irregular', 'Mass_Shape_lobular', 'Mass_Shape_polygonal', 'Mass_Shape_round/oval',
    
    'Mass_Margin_spiculated', 'Mass_Margin_microlobulated', 'Mass_Margin_amorphous',
    'Mass_Margin_circumscribed', 'Mass_Margin_obscured',

    'Mass_Density_low_density', 'Mass_Density_equal_density', 'Mass_Density_high_density',
    'Mass_Margin_indistinct',
    
    'Calcification_Location_L', 'Calcification_Location_M', 'Calcification_Location_M-L',
    'Calcification_Location_S', 'Calcification_Location_U', 'Calcification_Location_U-M',
    'Calcification_Location_Whole area',

    'Calcification_Morphology_pleomorphic', 'Calcification_Morphology_amorphous',
    'Calcification_Morphology_fine_and_linear', 'Calcification_Morphology_small_round',
    
    'Calcification_Distribution_grouped', 'Calcification_Distribution_linear',
    'Calcification_Distribution_regional', 'Calcification_Distribution_segmental',
    'Calcification_Distribution_segmental/linear'
]

desired_order = non_predictive_columns + predictive_columns_order
df_encoded = df_encoded[desired_order]

desired_order = non_predictive_columns + predictive_columns_order
df_encoded = df_encoded[desired_order]
# %%

# # transform patient lvl dataframe to lesion lvl dataframe
# def split_rows(df):
#     new_rows = []
#     for _, row in df.iterrows():
#         points = row['points']
#         labels = row['label']
        
#         if len(points) == 0 and len(labels) == 0:
#             new_rows.append(row)
#         else:
#             for point, label in zip(points, labels):
#                 new_row = row.copy()
#                 new_row['points'] = point
#                 new_row['label'] = label
#                 new_rows.append(new_row)
    
#     return pd.DataFrame(new_rows)

# def fill_if_missing(x):
#     if x is None or (isinstance(x, float) and pd.isna(x)):
#         return ["None"]
#     return x

# df_encoded = df_encoded.applymap(fill_if_missing)
# df_encoded = split_rows(df_encoded)
# %%
df_encoded = df_encoded.dropna(subset=["subtype"])
# %%

# save the final dataframe to pkl file
output_path = config['dir']['df']
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_encoded.to_pickle(output_path)
logger.info(f"Dataframe saved to {output_path}")
# %%