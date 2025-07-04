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
from model import IntermediateFusionModel, DecisionLevelLateFusionModel
from net_utils import train, validate, test, EarlyStopping, deactivate_batchnorm
from torch.utils.data import DataLoader
from dataset import MultimodalDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

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
# parser.add_argument("--fold", type=int, default=None)
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# current_fold = args.fold if args.fold else config['fold']
current_fold = config['training_plan']['parameters']['fold']

if config["neptune"]:
    run = neptune.init_run(project="ProjektMMG/multimodal-fusion",)
    run["sys/group_tags"].add(config["modality"])
    run["config"] = config
    run['train/current_fold'] = current_fold
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

test_size = 0.15
unique_ids = df_encoded["ID"].unique()
id_classification_map = df_encoded.groupby("ID")["subtype"].first()

train_val_ids, test_ids = train_test_split(
    unique_ids, test_size=test_size, stratify=id_classification_map[unique_ids], random_state=seed)

test_df = df_encoded[df_encoded["ID"].isin(test_ids)]
train_val_df = df_encoded[df_encoded["ID"].isin(train_val_ids)]

SPLITS = 10
if run:
    run['train/splits'] = SPLITS

kf = StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=seed)
folds = list(kf.split(train_val_ids, id_classification_map[train_val_ids]))

print(f"Train/Val set size: {len(train_val_df)}")
print(f"Test set size: {len(test_df)}")
print(f"Number of folds: {len(folds)}")

# %%
# TODO
# implement transforms for tabular data
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5)
])
modality = config['modality']

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    if fold_idx != current_fold:
        continue
    train_ids = train_val_ids[train_idx]
    val_ids = train_val_ids[val_idx]

    if run:
        run[f'train/ids/fold_{fold_idx}'] = train_ids.tolist()
        run[f'val/ids/fold_{fold_idx}'] = val_ids.tolist()
        run['test/ids'] = test_ids.tolist()

    train_df = df_encoded[df_encoded["ID"].isin(train_ids)]
    val_df = df_encoded[df_encoded["ID"].isin(val_ids)]

train_dataset = MultimodalDataset(train_df, subset='train', modality=modality, transform=train_transforms)
val_dataset = MultimodalDataset(val_df, subset='val', modality=modality, transform=None)
test_dataset = MultimodalDataset(test_df, subset='test', modality=modality, transform=None)

# %%
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

dataloaders = {'train': train_loader,
               'val': val_loader,
               'test': test_loader}
# %%
if config['training_plan']['fusion'] == 'intermediate':
    model = IntermediateFusionModel(modality=modality, device=device)
elif config['training_plan']['fusion'] == 'late':
    model = DecisionLevelLateFusionModel(modality=modality, out_dim=4, device=device)
model.apply(deactivate_batchnorm)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
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


if config['training_plan']['fusion'] == 'intermediate':
    model = IntermediateFusionModel(modality=modality, device=device)
elif config['training_plan']['fusion'] == 'late':
    model = DecisionLevelLateFusionModel(modality=modality, out_dim=4, device=device)
model.apply(deactivate_batchnorm)
model.load_state_dict(torch.load(model_name))
model.to(device)
test(model, dataloaders['test'], device, run)
if run is not None:
    run.stop()
# %%