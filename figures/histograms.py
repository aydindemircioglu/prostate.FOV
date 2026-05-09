#!/usr/bin/python3
import os
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    centers = sorted(glob("../data/test/*/"))
    dfs = []
    for c in centers:
        p = os.path.join(c, "stats.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if "true_slice" not in df.columns or "iou_cor" not in df.columns:
            continue
        df["_site"] = os.path.basename(c.rstrip("/"))
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No valid centers found.")

    n = len(dfs)
    fig, axs = plt.subplots(3, n, figsize=(6 * n + 1, 13))
    if n == 1:
        axs = axs.reshape(3, 1)

    all_data = pd.concat(dfs)
    angle_bins = np.linspace(all_data["angle_diff_cor"].min(), all_data["angle_diff_cor"].max(), 21)
    iou_cor_bins = np.linspace(all_data["iou_cor"].min(), all_data["iou_cor"].max(), 21)
    iou_tra_bins = np.linspace(all_data["iou_tra"].min(), all_data["iou_tra"].max(), 21)

    for j, df in enumerate(dfs):
        site = df["_site"].iloc[0]
        sns.histplot(df["angle_diff_cor"], bins=angle_bins, kde=True, ax=axs[0, j])
        axs[0, j].set_xlim([-30, 30])
        axs[0, j].set_xlabel("Angle Difference", fontsize=15)
        axs[0, j].set_ylabel("Frequency", fontsize=15)
        axs[0, j].tick_params(axis="both", labelsize=15)
        axs[0, j].set_title(site, fontsize=15)

        sns.histplot(df["iou_cor"], bins=iou_cor_bins, kde=True, ax=axs[1, j])
        axs[1, j].set_xlim([all_data["iou_cor"].min(), all_data["iou_cor"].max()])
        axs[1, j].set_xlabel("IoU (coronal)", fontsize=15)
        axs[1, j].set_ylabel("Frequency", fontsize=15)
        axs[1, j].tick_params(axis="both", labelsize=15)
        axs[1, j].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axs[1, j].set_title(site, fontsize=15)

        sns.histplot(df["iou_tra"], bins=iou_tra_bins, kde=True, ax=axs[2, j])
        axs[2, j].set_xlim([all_data["iou_tra"].min(), all_data["iou_tra"].max()])
        axs[2, j].set_xlabel("IoU (axial)", fontsize=15)
        axs[2, j].set_ylabel("Frequency", fontsize=15)
        axs[2, j].tick_params(axis="both", labelsize=15)
        axs[2, j].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axs[2, j].set_title(site, fontsize=15)

    row_labels = ["Histograms of angle differences", "Histograms of IoU (coronal)", "Histograms of IoU (axial)"]
    row_y = [0.91, 0.614, 0.32]
    for label, y in zip(row_labels, row_y):
        fig.text(0.5, y, label, ha="center", fontsize=16)

    plt.subplots_adjust(wspace=0.40, hspace=0.77)
    plt.savefig("./histo_plots.png", dpi=250, bbox_inches="tight")
    plt.savefig("./histo_plots.tiff", dpi=250, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
