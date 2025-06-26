 # %% IMPORTS AND SETTINGS
import os
import yaml
import uuid
import torch
import typing
import neptune
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import albumentations as A
# import matplotlib.pyplot as plt
from model import EarlyFusionModel
from net_utils import train, validate, test, EarlyStopping, deactivate_batchnorm
from torch.utils.data import DataLoader
from dataset import MultimodalDataset
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)
logger_radiomics = logging.getLogger("radiomics")
logger_radiomics.setLevel(logging.ERROR)

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

if config["neptune"]:
    run = neptune.init_run(project="ProjektMMG/multimodal-fusion",)
    run["sys/group_tags"].add(["baseline-img-only"])
    # run["sys/group_tags"].add(["baseline-img-clinical"])
    # run["sys/group_tags"].add(["baseline-img-clinical-radiomics"])
    # run["sys/group_tags"].add(["baseline-img-clinical-radiomics-metalesion"])
    
    run["config"] = config
else:
    run = None

# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = config['device'] if torch.cuda.is_available() else 'cpu'

# LOAD DATA -------------------------------------------------------------
df_encoded = pd.read_pickle(config['dir']['df'])
# %%
fixed_columns = df_encoded.columns[:6]
additional_fixed_columns = ['LeftRight_L', 'LeftRight_R']
remaining_columns = [col for col in df_encoded.columns[6:] if col not in additional_fixed_columns]
remaining_columns = sorted(remaining_columns)
df_encoded = df_encoded[list(fixed_columns) + additional_fixed_columns + remaining_columns]

# %%

# TODO 
# implement cross-validation

unique_ids = df_encoded["ID"].unique()
id_classification_map = df_encoded.groupby("ID")["classification"].first()

train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, stratify=id_classification_map[unique_ids], random_state=seed)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, stratify=id_classification_map[temp_ids], random_state=seed)

train_df = df_encoded[df_encoded["ID"].isin(train_ids)]
val_df = df_encoded[df_encoded["ID"].isin(val_ids)]
test_df = df_encoded[df_encoded["ID"].isin(test_ids)]

print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# %%
# TODO
# implement transforms for tabular data
train_transforms = A.Compose([
    A.Flip(p=0.5),
    A.VerticalFlip(p=0.5)
])
train_dataset = MultimodalDataset(train_df, subset='train', transform=train_transforms)
val_dataset = MultimodalDataset(val_df, subset='val', transform=None)
test_dataset = MultimodalDataset(test_df, subset='test', transform=None)

# %%
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

dataloaders = {'train': train_loader,
               'val': val_loader,
               'test': test_loader}
# %%
modality = config['modality']
model = EarlyFusionModel(modality=modality, device=device)
model.apply(deactivate_batchnorm)
model.to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# %%
early_stopping = EarlyStopping(patience=config['training_plan']['parameters']['patience'], neptune_run=run)

for epoch in range(1, config['training_plan']['parameters']['epochs'] + 1):
    train(model, dataloaders['train'], criterion, optimizer, device, run, epoch)
    val_loss = validate(model, dataloaders['val'], criterion, device, run, epoch)
    if early_stopping(val_loss, model):
        print(f"Early stopping at epoch {epoch}")
        break
model_name = uuid.uuid4().hex
model_name = os.path.join(config['model_path'], model_name)
if not os.path.exists(config['model_path']):
    os.makedirs(config['model_path'])
torch.save(early_stopping.get_best_model_state(), model_name)
if run is not None:
    run["best_model_path"].log(model_name)


model = EarlyFusionModel(modality, device=device)
model.apply(deactivate_batchnorm)
model.load_state_dict(torch.load(model_name))
model.to(device)
test(model, dataloaders['test'], device, run)
if run is not None:
    run.stop()



# %%
# # IDs with wrong laterality in annotations or DICOM files (checked manually)
# def visualize_sample(dataset, index):
#     sample = dataset[index]
#     image = sample['image']
#     mask = sample['mask']

#     fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
#     ax[0].imshow(image.squeeze(), cmap='gray')
#     ax[0].set_title("Image")
#     ax[0].axis('off')
    
#     ax[1].imshow(mask.squeeze(), cmap='jet', alpha=0.6)
#     ax[1].set_title("Mask")
#     ax[1].axis('off')
    
#     plt.tight_layout()
#     plt.show()

# i = 0
# for index, sample in enumerate(train_dataset):
#     if i >= 5:  # Limit to first 5 samples
#         break
#     i += 1
#     print(f"Patient ID: {sample['patient_id']}")
#     print(f"Laterality L?: {sample['tabular'][1]}")
#     visualize_sample(train_dataset, index)