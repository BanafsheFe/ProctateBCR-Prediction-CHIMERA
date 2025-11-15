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
#from util_functions.Prostate_DataGenerator import Survival_MRI_Generator,US_MRI_Generator, collate_prostate_position_CS, getData
from ProViCNet.util_functions.utils_weighted import set_seed
from ProViCNet.util_functions.train_functions import getPatchTokens
from util_functions.train_functions import tensor_shuffle, getPatchTokens, OneBatchTraining_fusion
from ProViCNet.util_functions.Prostate_DataGenerator import US_MRI_Generator, Survival_MRI_Generator,collate_prostate_position_CS#,Survival_3MRI_Generator
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from types import SimpleNamespace
from sklearn.model_selection import train_test_split
# %%
from data.Prostate_BCR_DataGenerator import Survival_3MRI_Generator
from models.Survival_heads import FullModelTokens, FullModelTokens_tiny, FullModelTokens_MeanFusion,EarlyStopping
from my_utils.Helpers import *
from models.Loss_functions import cox_loss, soft_cindex_loss
from my_utils.model_utils import *

# %%
import weave
import wandb
from types import SimpleNamespace



def load_weight_from_url(url, device):
    """
    Downloads the weight file from Hugging Face Hub and loads it.
    Assumes URL format: "https://huggingface.co/{repo_id}/resolve/main/{filename}"
    """
    parts = url.split('/')
    repo_id = f"{parts[3]}/{parts[4]}"
    filename = parts[-1]
    weight_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print("weight_path",weight_path)
    return torch.load(weight_path, map_location=device)

def main(args):


    # Load YAML config
    with open(args.config_file) as f:
        args.config = yaml.load(f, Loader=yaml.FullLoader)
        config = args.config

    print(f"Configurations: {config}")
    args.lr =0.001
    args.epochs =100
    args.num_workers=0
    args.Bag_batch_size=2
    args.img_size = 448
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
 
    num_epochs = args.epochs
    flag_clinical_data_use=True
    single_scan =args.single_scan

    survival_head_name =args.survival_head_name
    experiment_name =f"mpMRI_BCR_prediction__{survival_head_name}_sinlgeloss_cox_clinical"

    if single_scan:
        experiment_name =f"mpMRI_BCR_prediction__{survival_head_name}_sinlgeloss_cox_clinical_single"



    NUM_FOLDS =5
    for fold_idx in range(0,NUM_FOLDS):
        print(f"\n=== Fold {fold_idx} ==={experiment_name}===")
        
        best_cindex=0

        best_model_path = f"./weights/{experiment_name}/full_model_fold{fold_idx}_best.pt"
        latest_model_path = f"./weights/{experiment_name}/latest_model_fold{fold_idx}.pt"
        os.makedirs(f"./weights/{experiment_name}/", exist_ok=True)

        # prepare train/val split
        trainval_ids = df[df["fold"] != fold_idx]["patient_id"].astype(int).tolist()
        train_ids, val_ids = train_test_split(trainval_ids, test_size=0.16, random_state=42, shuffle=True  )
        TrainDataset = Dataset_df[Dataset_df["patient_id"].isin(train_ids)]
        ValidDataset = Dataset_df[Dataset_df["patient_id"].isin(val_ids)]
        print(f"Train: {len(TrainDataset)}, Val: {len(ValidDataset)}")

        if flag_clinical_data_use:
            X_train=TrainDataset[clinical_data_cols]
            X_val = ValidDataset[clinical_data_cols]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
        else: 
            X_train_scaled=None
            X_val_scaled =None
        train_loader, val_loader = build_dataloaders(TrainDataset, ValidDataset, batch_size_train=3, batch_size_val=4, img_size=args.img_size, train_clinical=X_train_scaled, val_clinical=X_val_scaled)
        
        full_model = create_survival_model(survival_head_name, embedding_size, len(clinical_data_cols), args.device)
        optimizer = optim.Adam(full_model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-7, verbose=True)
        early_stopper = EarlyStopping(patience=10, min_epochs=20,save_path=f"./{experiment_name}/best_fold_es_{fold_idx}.pt")
        

        wandb.init(
        project=experiment_name,
        name=f"fold_{fold_idx}",
        reinit=True  # <--- important when doing multiple folds!
        )
        wandb.watch(full_model, log="all", log_freq=50)


        for epoch in range(num_epochs):

            if single_scan:
                train_loss, _ = run_one_epoch_single_radio(full_model, MODELs,train_loader, optimizer=optimizer, device=args.device, train=True, args=args)
                val_loss, c_index = run_one_epoch_single_radio(full_model, MODELs,val_loader, optimizer=None, device=args.device, train=False, args=args)
            else:   
                train_loss, _ = run_one_epoch(full_model, MODELs,train_loader, optimizer=optimizer, device=args.device, train=True, args=args)
                val_loss, c_index = run_one_epoch(full_model, MODELs,val_loader, optimizer=None, device=args.device, train=False, args=args)
                
            scheduler.step(c_index)
            early_stopper(c_index, model=full_model, epoch=epoch)
            if early_stopper.early_stop:
                break

            c_index_final = max(c_index, 1-c_index)

            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}, C-index: {c_index:.4f}")
            # ---- Save best model based on C-index ----
            if c_index >= best_cindex:
                best_cindex = c_index
                torch.save(full_model.state_dict(), best_model_path)
                print(f"--> New best model saved for fold {fold_idx} with C-index: {best_cindex:.4f}")
            torch.save(full_model.state_dict(), latest_model_path)


            # ---- wandb Logging --------------------------------
            wandb.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/c_index": c_index,
                "val/c_index_final": c_index_final,
                "epoch": epoch
            })

        # 3. Finish
        wandb.finish()

        pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script for ProViCNet survival model.")

    # -------------------------
    # Model settings
    # -------------------------
    parser.add_argument("--ModelName", type=str, default="ProViCNet")
    parser.add_argument("--vit_backbone", type=str, default="dinov2_s_reg")
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--nClass", type=int, default=4)
    parser.add_argument("--nChannel", type=int, default=3)
    parser.add_argument("--contrastive", type=bool, default=True)

    # -------------------------
    # Training
    # -------------------------
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--Bag_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--small_batchsize", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)

    # -------------------------
    # Path / config
    # -------------------------
    parser.add_argument("--config_file", type=str, default="configs/config_infer_MRI.yaml")
    parser.add_argument("--save_folder", type=str, default="results_ProViCNet/")
    parser.add_argument("--visualization_folder", type=str, default="visualization_ProViCNet/")

    # -------------------------
    # Additional flags
    # -------------------------
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--only_csPCa", type=bool, default=False)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--single_scan", type=bool, default=False)
    parser.add_argument("--survival_head_name", type=str, default='tiny')

    args = parser.parse_args()

    # -------------------------
    # Device configuration
    # -------------------------
    if torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.cuda_device}")
        print(f"Using GPU: cuda:{args.cuda_device}")
    else:
        args.device = torch.device("cpu")
        print("Using CPU")

    # -------------------------
    # Load YAML config file
    # -------------------------
    with open(args.config_file) as f:
        args.config = yaml.safe_load(f)
        print("Loaded configuration:", args.config)

    # -------------------------
    # Run Training
    # -------------------------
    main(args)
