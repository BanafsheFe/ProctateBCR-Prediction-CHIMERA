#%%

import matplotlib.pyplot as plt
import torch
import argparse
import os
import numpy as np
import torch
import yaml
import json
from pathlib import Path
import pandas as pd


def match_slice_length(img, target_len):
    """
    img: [B, S, C, H, W]
    target_len: int
    """
    Im_shape = img.shape
    #B, S, C, H, W = img.shape
    
    B, S =Im_shape[:2]
    if S == target_len:
        return img
    elif S < target_len:
        diff = target_len - S
        # pad equally on both sides if possible
        pad_front = diff // 2
        pad_back = diff - pad_front
        front_slices = img[:, :1].repeat(1, pad_front, 1, 1, 1)
        back_slices  = img[:, -1:].repeat(1, pad_back, 1, 1, 1)
        img = torch.cat([front_slices, img, back_slices], dim=1)
    else:
        # too many slices → center crop
        start = (S - target_len) // 2
        img = img[:, start:start+target_len]

    return img

def filter_slices_by_label(self,gland, min_label_fraction=0.005):
    """
    Keep only slices where label (cancer) covers at least min_label_fraction of the image.
    Args:
        image: np.array [num_slices, H, W]
        gland: np.array [num_slices, H, W]
        cancer: np.array [num_slices, H, W]
        min_label_fraction: minimum label area / total pixels per slice
    Returns:
        filtered_image, filtered_gland, filtered_cancer
    """
    keep_indices = []
    H, W = gland.shape[1:]
    total_pixels = H * W

    for i in range(gland.shape[0]):
        label_fraction = np.count_nonzero(gland[i]) / total_pixels
        if label_fraction >= min_label_fraction:
            keep_indices.append(i)

    # If all slices are empty (no tumor), keep at least one central slice
    if len(keep_indices) == 0:
        keep_indices = [gland.shape[0] // 2]


    return keep_indices





def visualize_slices(
    Image_T2, Image_ADC, Image_DWI, gland_label=None, cancer_label=None,
    slice_idx=0, channel_idx=0, rescale=False
):
    """
    Visualize a slice from T2, ADC, and DWI tensors side by side,
    along with optional gland and cancer labels.
    
    Args:
        Image_T2, Image_ADC, Image_DWI: [B, num_slices, C, H, W]
        gland_label, cancer_label: [B, num_slices, 1, H, W] (optional)
        slice_idx: index of the slice to visualize
        channel_idx: index of the channel
        rescale: whether to normalize slices for display
    """
    
    # Select first batch
    T2_slice = Image_T2[0, slice_idx, channel_idx].cpu().numpy()
    ADC_slice = Image_ADC[0, slice_idx, channel_idx].cpu().numpy()
    DWI_slice = Image_DWI[0, slice_idx, channel_idx].cpu().numpy()

    # Normalize to [0, 1] for display
    if rescale:
        normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        T2_slice, ADC_slice, DWI_slice = map(normalize, [T2_slice, ADC_slice, DWI_slice])

    # Prepare subplots
    n_cols = 3 + int(gland_label is not None) + int(cancer_label is not None)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    # Display MRI slices
    axes[0].imshow(T2_slice, cmap='gray')
    axes[0].set_title(f"T2 Slice {slice_idx}")
    axes[1].imshow(ADC_slice, cmap='gray')
    axes[1].set_title(f"ADC Slice {slice_idx}")
    axes[2].imshow(DWI_slice, cmap='gray')
    axes[2].set_title(f"DWI Slice {slice_idx}")

    # Display gland label (if provided)
    col = 3
    if gland_label is not None:
        gland = gland_label[0, slice_idx].cpu().numpy()
        axes[col].imshow(T2_slice, cmap='gray')
        axes[col].imshow(gland, cmap='Reds', alpha=0.5)
        axes[col].set_title("Gland Mask")
        col += 1

    # Display cancer label (if provided)
    if cancer_label is not None:
        cancer = cancer_label[0, slice_idx, 0].cpu().numpy()
        axes[col].imshow(T2_slice, cmap='gray')
        axes[col].imshow(cancer, cmap='Blues', alpha=0.5)
        axes[col].set_title("Cancer Mask")

    # Hide axes
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


from lifelines import KaplanMeierFitter, CoxPHFitter

def plot_km_by_median(df, risk_col="risk", duration_col="duration", event_col="event", title=None, save_path=None):
    median = df[risk_col].median()
    df["risk_group"] = (df[risk_col] > median).astype(int)
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(6,4))
    for label, grp in df.groupby("risk_group"):
        kmf.fit(grp[duration_col], grp[event_col], label=f"risk_group={label} (n={len(grp)})")
        kmf.plot(ci_show=True)
    plt.title(title or "Kaplan-Meier by median risk")
    plt.xlabel("Time (months)")
    plt.ylabel("Survival probability")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
