#%%
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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Import model, data generator, and utility functions from ProViCNet
from ProViCNet.ModelArchitectures.Models import GetModel
from ProViCNet.ModelArchitectures.ProViCNet.ProViCNet import FusionModalities, load_partial_weights
from ProViCNet.util_functions.utils_weighted import set_seed
from ProViCNet.util_functions.train_functions import getPatchTokens
from util_functions.train_functions import tensor_shuffle, getPatchTokens, OneBatchTraining_fusion
from ProViCNet.util_functions.Prostate_DataGenerator import US_MRI_Generator, Survival_MRI_Generator,collate_prostate_position_CS#,Survival_3MRI_Generator
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from types import SimpleNamespace
from sklearn.model_selection import train_test_split

from data.Prostate_BCR_DataGenerator import Survival_3MRI_Generator
from models.Survival_heads import FullModelTokens, FullModelTokens_tiny, FullModelTokens_MeanFusion,EarlyStopping
from my_utils.Helpers import *
from models.Loss_functions import cox_loss, soft_cindex_loss
from my_utils.model_utils import *

from types import SimpleNamespace
# %%

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler



EXPERIMENTS = {

    "mpMRI_BCR_prediction__tiny": {
        "model_class": FullModelTokens_tiny,
        "model_class_name":"tiny",
        "experiment_name": "mpMRI_BCR_prediction__tiny",
        "modalities": ["ADC", "DWI", "T2"],
        "use_clinical": False,
        "loss": "cox",
    },



    "mpMRI_BCR_prediction__tiny_sinlgeloss_cox": {
        "model_class": FullModelTokens_tiny,
        "model_class_name":"tiny",
        "experiment_name": "mpMRI_BCR_prediction__tiny_sinlgeloss_cox",
        "modalities": ["ADC", "DWI", "T2"],
        "use_clinical": False,
        "loss": "cox",
    },


    "mpMRI_BCR_prediction__tiny_sinlgeloss_cox_clinical": {
        "model_class": FullModelTokens_tiny,
        "model_class_name":"tiny",
        "experiment_name": "mpMRI_BCR_prediction__tiny_sinlgeloss_cox_clinical",
        "modalities": ["ADC", "DWI", "T2"],
        "use_clinical": True,
        "loss": "cox",
    },

    "mpMRI_BCR_prediction__tiny_sinlgeloss_cox_clinical_1MRI": {
        "model_class": FullModelTokens_MeanFusion,
        "model_class_name":"MeanFusion",
        "experiment_name": "mpMRI_BCR_prediction__tiny_sinlgeloss_cox_clinical_1MRI",
        "modalities": ["T2"],
        "use_clinical": True,
        "loss": "cox",
    },


    "mpMRI_BCR_prediction__Full": {
        "model_class": FullModelTokens,
        "model_class_name":"Full",
        "experiment_name": "mpMRI_BCR_prediction__Full",
        "modalities": ["ADC", "DWI", "T2"],
        "use_clinical": False,
        "loss": "cox",
    },


    "mpMRI_BCR_prediction__Full_sinlgeloss_cox_clinical": {
        "model_class": FullModelTokens,
        "model_class_name":"Full",
        "experiment_name": "mpMRI_BCR_prediction__Full_sinlgeloss_cox_clinical",
        "modalities": ["ADC", "DWI", "T2"],
        "use_clinical": True,
        "loss": "cox",
    },
}



#%%
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

# Optional: set seed if you have a set_seed function
set_seed(42)
# %%

# ============================================================
# 1. PATH SETUP
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent
CODE_ROOT =Path(__file__).resolve().parent
# Load config
with open(os.path.join(CODE_ROOT , "configs/config_train_survival.yaml"), "r") as f:
    config = yaml.safe_load(f)

# Build paths
radiology_data_path = PROJECT_ROOT / config["paths"]["radiology_data_path"]
clinical_data_path  = PROJECT_ROOT / config["paths"]["clinical_folder"]
mpMRI_folder        = config["paths"]["mpMRI_folder"]
mask_folder         = config["paths"]["mask_folder"]

# ============================================================
# 2. LOAD FOLD SPLIT
# ============================================================

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
subjects = df["patient_id"].astype(int).tolist()  # all subjects
val_ids = df[df["fold"] == fold_idx]["patient_id"].astype(int).tolist()  # validation subjects

# Define train_ids as all subjects not in val_ids
train_ids = [int(sbj) for sbj in subjects if sbj not in val_ids]

# Build Dataset dictionary
Dataset = {
    sbj: {
        "T2":     os.path.join(radiology_data_path, mpMRI, str(sbj), f"{sbj}{config['file_extensions']['T2']}"),
        "ADC":    os.path.join(radiology_data_path, mpMRI, str(sbj), f"{sbj}{config['file_extensions']['ADC']}"),
        "DWI":    os.path.join(radiology_data_path, mpMRI, str(sbj), f"{sbj}{config['file_extensions']['DWI']}"),
        "Gland":  os.path.join(radiology_data_path, mask, f"{sbj}{config['file_extensions']['Gland']}"),
        "Cancer": os.path.join(radiology_data_path, mask, f"{sbj}{config['file_extensions']['Cancer']}"),
        "clinical": os.path.join(clinical_data_path, f"{sbj}.json"),
        "fold": int(df.loc[df["patient_id"] == sbj, "fold"].values[0])
    }
    for sbj in subjects
}

#%%

# Convert to DataFrame
Dataset_df = pd.DataFrame.from_dict(Dataset, orient="index")
Dataset_df["patient_id"] = Dataset_df.index.astype(int)

# ============================================================
# 4. CLINICAL DATA PROCESSING
# ============================================================
#presugical features
safe_clinical_features = ['ISUP', 'primary_gleason', 'secondary_gleason', 
                        'pre_operative_PSA', 'age_at_prostatectomy', 'earlier_therapy']

output_cols = ["event", "duration"]

clinical_df_path = PROJECT_ROOT / "Multimodal-Quiz/clinical_df.csv"
clinical_df = pd.read_csv(clinical_df_path)
clinical_df = clinical_df[["patient_id"] + safe_clinical_features + output_cols]

# Process categorical column
clinical_df = pd.get_dummies(clinical_df, columns=['earlier_therapy'])

# Numeric features for scaling
numeric_cols = ['ISUP', 'primary_gleason', 'secondary_gleason',
                'pre_operative_PSA', 'age_at_prostatectomy']

# Scale numeric values
scaler = StandardScaler()
clinical_df[numeric_cols] = scaler.fit_transform(clinical_df[numeric_cols])

# Updated column list
clinical_data_cols = numeric_cols + [ 'earlier_therapy_none',    'earlier_therapy_radiotherapy + cryotherapy',
                                    'earlier_therapy_radiotherapy + hormones',    'earlier_therapy_unknown']

# Ensure IDs match type
clinical_df["patient_id"] = clinical_df["patient_id"].astype(int)
Dataset_df["patient_id"] = Dataset_df["patient_id"].astype(int)

# Merge Dataset + clinical metadata
Dataset_df = Dataset_df.merge(clinical_df, on="patient_id", how="left")



###############################################################
# 2. Backbone Model Loading
###############################################################
# Pretrained weights for each modality model
# Load individual modality models (T2, ADC, DWI)
weights_path ="./weights/models--pimed--ProViCNet/snapshots/f9702f7cdb95eb43e2edb1b90148f0c0358a7757/"

set_seed(42)
MODELs = dict()

for Sequence in ['T2', 'ADC', 'DWI']:
    model = GetModel( args.ModelName, args.nClass, args.nChannel, args.img_size, vit_backbone=args.vit_backbone, contrastive=args.contrastive    )
    model = model.to(args.device)
    #state_dict = load_weight_from_url(args.config['model_weights'][Sequence], args.device)
    weight_path = f"{weights_path}/{Sequence}_best.pth"
    if os.path.exists(weight_path):
        # load weights
        state_dict=torch.load(weight_path, map_location=args.device)
    else: 
        state_dict = load_weight_from_url(args.config['model_weights'][Sequence], args.device)


    model.load_state_dict(state_dict, strict=True)
    model.eval()
    MODELs[Sequence] = model

embedding_size=MODELs['T2'].embedding_size




# %%



device = args.device

summary_all =[]
for experiment_name in EXPERIMENTS:
    clinical_features_dim = len(clinical_data_cols)

    
    NUM_FOLDS =5
    c_index_experiment =[]
    out_df_experiment = pd.DataFrame(columns=["Fold", "c_index_experiment", "HR"])

    experiment_results = []

    for fold_idx in range(0,NUM_FOLDS):
        print(f"\n=== Running Fold {fold_idx} ===")


        survival_head_name = EXPERIMENTS[experiment_name]["model_class_name"]
        clinical_usage =EXPERIMENTS[experiment_name]["use_clinical"]
        if clinical_usage:
            clinical_features_dim = len(clinical_data_cols)
            if experiment_name=="mpMRI_BCR_prediction__Full_sinlgeloss_cox_clinical":
                clinical_data_cols =clinical_data_cols[0:6]
        else:
            clinical_features_dim=0

        full_model = create_survival_model(survival_head_name, embedding_size, clinical_features_dim, args.device)



        test_ids = df[df["fold"] == fold_idx]["patient_id"].astype(int).tolist()
        train_ids = df[df["fold"] != fold_idx]["patient_id"].astype(int).tolist()

        TrainDataset = Dataset_df[Dataset_df["patient_id"].isin(train_ids)]
        TestDataset = Dataset_df[Dataset_df["patient_id"].isin(test_ids)]

        X_train=TrainDataset[clinical_data_cols]
        X_test = TestDataset[clinical_data_cols]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if not clinical_usage:
            X_test_scaled=None

        print(f"Train: {len(TrainDataset)}, Val: {len(TestDataset)}")
        best_model_path = f"./wieghts/{experiment_name}/full_model_fold{fold_idx}_best.pth"
        latest_model_path = f"./wieghts/{experiment_name}/latest_model_fold{fold_idx}.pt"
        result_path =f"./results/{experiment_name}/"
        os.makedirs(result_path, exist_ok=True)

        exp_cfg = EXPERIMENTS[experiment_name]
        best_model_path, latest_model_path = get_weight_paths(exp_cfg["experiment_name"], fold_idx)





        TEST_GENERATOR = Survival_3MRI_Generator(
            imageFileName=list(TestDataset['T2']),
            imageFileName2=list(TestDataset['ADC']),
            imageFileName3=list(TestDataset['DWI']),
            clinical_data=X_test_scaled,
            glandFileName=list(TestDataset['Gland']),
            cancerFileName=list(TestDataset['Cancer']),
            modality='MRI',
            Image_Only=False,
            img_size=args.img_size,
            event=list(TestDataset['event']),
            duration=list(TestDataset['duration']),
        )
        TEST_DATALOADER= DataLoader(TEST_GENERATOR, batch_size=4, shuffle=False)


        best_model_path = f"/{experiment_name}/full_model_fold{fold_idx}_best.pth"
        if not os.path.exists(best_model_path):
            best_model_path = latest_model_path  # fallback


        exp_cfg = EXPERIMENTS[experiment_name]
        best_model_path, latest_model_path = get_weight_paths(exp_cfg["experiment_name"], fold_idx)




        # 1) load trained model (use same constructor & kwargs used when creating full_model)
        model_for_infer = full_model
        model_for_infer = model_for_infer.to(device)

        if os.path.exists(best_model_path):
        # load weights
            model_for_infer.load_state_dict(torch.load(best_model_path, map_location=device))
        else: 
                blob_url = f"https://huggingface.co/BanafsheFelfeliyan/ProctateBCR-Prediction-CHIMERA/blob/main/weights/{experiment_name}/best_fold_es_{fold_idx}.pt"
                state_dict = load_hf_weight_from_blob(blob_url, device)
                model.load_state_dict(state_dict, strict=True)


        model_for_infer.eval()

        results = infer_on_dataloader(model_for_infer, MODELs, TEST_DATALOADER, device,args)

        # 3) compute concordance index
        c_index = concordance_index(results["duration"], -results["risk"], results["event"])
        
        c_index = max(c_index, 1-c_index)
        print(f"[Fold {fold_idx}] Validation C-index = {c_index:.4f}")

        # 4) assemble dataframe and save CSV
        out_df = pd.DataFrame({
            "duration": results["duration"],
            "event": results["event"],
            "risk": results["risk"]
        })

        csv_path = os.path.join(f"{result_path}", f"fold{fold_idx}_predictions.csv")
        out_df.to_csv(csv_path, index=False)
        print("Saved predictions to", csv_path)

        save_path=f"{result_path}/KM by median risk (experiment{experiment_name} fold {fold_idx})"
        plot_km_by_median(out_df, title=f"Fold {fold_idx} KM by median risk (c-index={c_index:.3f})",save_path=save_path)

        # 6) CoxPH using continuous risk as covariate (get HR for 1-unit risk change)
        cox_df = out_df.copy()
        cox_df["duration_n"] = -1 * cox_df["duration"]
        cox_df = cox_df.rename(columns={"duration":"T", "event":"E"})
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="T", event_col="E")
        #print(cph.summary)

        HR = cph.summary.loc["risk", "exp(coef)"]
        HR=HR.round(3)
        print("Hazard Ratio:", HR.round(3))
        experiment_results.append({
            "experiment_name": experiment_name,
            "fold": fold_idx,
            "c_index": c_index.round(3),
            "HR": HR
        })


    out_df_experiment = pd.DataFrame(experiment_results)
    #out_df_experiment.to_csv(f"{result_path}/experiment_results.csv", index=False)

    # Summary statistics
    summary_df = out_df_experiment.groupby("experiment_name").agg(
        mean_c_index=("c_index", "mean"),
        std_c_index=("c_index", "std"),
        mean_HR=("HR", "mean"),
        std_HR=("HR", "std"),
        n_folds=("fold", "count")
    ).reset_index()


    numeric_cols = ["mean_c_index", "std_c_index", "mean_HR", "std_HR"]
    summary_df[numeric_cols] = summary_df[numeric_cols].round(3)
    summary_all.append(summary_df)
    #summary_df.to_csv(f"{result_path}/experiment_summary.csv", index=False)

    # Optional combined file
    combined_df = pd.concat(
        [out_df_experiment, summary_df],
        keys=["fold_results", "summary"]
    )
    combined_df.to_csv(f"{result_path}/experiment_all_results.csv")


# After looping over all experiments, concatenate into one DataFrame
final_summary_df = pd.concat(summary_all, axis=0, ignore_index=True)

# Save
final_summary_df.to_csv(f"./results/all_experiments_summary.csv", index=False)


# %%
