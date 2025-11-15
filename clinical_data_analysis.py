#%%
import json
import pandas as pd
import numpy as np
import os


import argparse
import os
import numpy as np
import torch
import yaml
import json
from pathlib import Path
import pandas as pd

from tqdm import tqdm
from huggingface_hub import hf_hub_download
import argparse

import torch.nn as nn
import torch.optim as optim




# %%

from types import SimpleNamespace

# Manually set arguments (same as your defaults or custom values)
args = SimpleNamespace(
    ModelName="ProViCNet",
    vit_backbone="dinov2_s_reg",
    img_size=448,
    nClass=4,
    nChannel=3,
    contrastive=True,
    cuda_device=0,
    only_csPCa=False,
    save_folder='results_ProViCNet/',
    visualization_folder='visualization_ProViCNet/',
    threshold=0.4,
    small_batchsize=16,
    config_file='configs/config_infer_MRI.yaml'
)

# Set device
args.device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")

# Load YAML config
with open(args.config_file) as f:
    args.config = yaml.load(f, Loader=yaml.FullLoader)
    config = args.config

print(f"Configurations: {config}")
args.lr =0.001
args.epochs =10
# Optional: set seed if you have a set_seed function
set_seed(42)
# %%

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
CODE_ROOT =Path(__file__).resolve().parent
# Load config
with open(os.path.join(CODE_ROOT , "configs/config_train_survival.yaml"), "r") as f:
    config = yaml.safe_load(f)

radiology_data_path =str(PROJECT_ROOT)+config["paths"]["radiology_data_path"]
clinical_data_path =str(PROJECT_ROOT)+config["paths"]["clinical_folder"]
mpMRI = config["paths"]["mpMRI_folder"]
mask = config["paths"]["mask_folder"]

# Load fold split
split_path =os.path.join(PROJECT_ROOT, config["SplitValidation"]["internal_split"])
split_path =str(PROJECT_ROOT)+config["SplitValidation"]["internal_split"]
df = pd.read_csv(split_path)


# Extract subject IDs for this fold
fold_idx = config["FOLD_IDX"]
subjects = df["patient_id"].astype(str).tolist()  # all subjects
val_ids = df[df["fold"] == fold_idx]["patient_id"].astype(str).tolist()  # validation subjects

# Define train_ids as all subjects not in val_ids
train_ids = [sbj for sbj in subjects if sbj not in val_ids]

# Build Dataset dictionary
Dataset = {
    sbj: {
        "T2":     os.path.join(radiology_data_path, mpMRI, sbj, f"{sbj}{config['file_extensions']['T2']}"),
        "ADC":    os.path.join(radiology_data_path, mpMRI, sbj, f"{sbj}{config['file_extensions']['ADC']}"),
        "DWI":    os.path.join(radiology_data_path, mpMRI, sbj, f"{sbj}{config['file_extensions']['DWI']}"),
        "Gland":  os.path.join(radiology_data_path, mask, f"{sbj}{config['file_extensions']['Gland']}"),
        "Cancer": os.path.join(radiology_data_path, mask, f"{sbj}{config['file_extensions']['Cancer']}"),
        "clinical": os.path.join(clinical_data_path, f"{sbj}.json"),
        "fold": int(df.loc[df["patient_id"].astype(str) == sbj, "fold"].values[0])
    }
    for sbj in subjects
}

# Convert Dataset dict to DataFrame for easy indexing
Dataset_df = pd.DataFrame.from_dict(Dataset, orient="index")

#%%
all_clinical_keys = set()
subject_ids = []

for sbj, row in Dataset_df.iterrows():
    clinical_path = row['clinical']
    subject_ids.append(sbj)
    
    # Read JSON file
    with open(clinical_path, 'r') as f:
        clinical_data = json.load(f)
    
    # Add keys to the set
    all_clinical_keys.update(clinical_data.keys())

print("All clinical features across subjects:")
print(all_clinical_keys)

#%%
subject_ids = []
events = []
durations = []
clinical_data_list = []

# Dynamically collect all keys
all_keys = {'age_at_prostatectomy', 'primary_gleason', 'capsular_penetration', 'secondary_gleason', 
            'BCR_PSA', 'positive_surgical_margins', 'ISUP', 'lymphovascular_invasion', 
            'invasion_seminal_vesicles', 'tertiary_gleason',  'pT_stage', 
            'pre_operative_PSA', 'earlier_therapy',  'positive_lymph_nodes'}

for sbj, row in Dataset_df.iterrows():
    clinical_path = row['clinical']
    subject_ids.append(sbj)
    
    with open(clinical_path, 'r') as f:
        clinical_json = json.load(f)
    
    # Event and duration
    if clinical_json.get("BCR", "0") == "1.0":
        event = 1
        duration = float(clinical_json.get("time_to_follow-up/BCR", np.nan))
    else:
        event = 0
        duration = float(clinical_json.get("time_to_follow-up/BCR", np.nan))

    events.append(event)
    durations.append(duration)
    
    # Collect all features dynamically, fill missing with NaN
    features_row = {key: clinical_json.get(key, np.nan) for key in all_keys}
    clinical_data_list.append(features_row)

# Compute 90th percentile of observed event durations (only events that happened)
event_durations = [d for d, e in zip(durations, events) if e == 1]
event_durations = [d for d, e in zip(durations, events) ]

max_duration = np.percentile(event_durations, 100).round(0)+2

# Replace NaNs (censored) with this 90th percentile value
durations = [d if not np.isnan(d) else max_duration for d in durations]

print(f"90th percentile max duration used for censored patients: {max_duration:.2f} months")

# Construct clinical DataFrame
clinical_df = pd.DataFrame(clinical_data_list)
clinical_df["patient_id"] = subject_ids
clinical_df["event"] = events
clinical_df["duration"] = durations

print(clinical_df.head())

# Save to CSV
clinical_df.to_csv(str(PROJECT_ROOT) + '/Multimodal-Quiz/clinical_df.csv', index=False)

# %%
