import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob


def add_border(img, pos, thickness):
    if pos == "H":
        img = np.hstack([
            255 * np.ones((img.shape[0], int(img.shape[1] * thickness), 3), dtype=np.uint8),
            img,
        ])
    if pos == "V":
        img = np.vstack([
            255 * np.ones((int(img.shape[0] * thickness), img.shape[1], 3), dtype=np.uint8),
            img,
        ])
    return img


def join_plots(result_dir):
    imA = cv2.imread(os.path.join(result_dir, "sensitivity_noise_plot.tiff"))
    imB = cv2.imread(os.path.join(result_dir, "sensitivity_contrast_plot.tiff"))
    imB = add_border(imB, "V", 0.05)
    imC = imB * 0 + 255
    imC[: imA.shape[0], : imA.shape[1], :] = imA
    imA = imC
    imgU = np.vstack([imA, imB])
    cv2.imwrite(
        os.path.join(result_dir, "Figure_Robustness.tiff"),
        imgU,
        params=(cv2.IMWRITE_TIFF_COMPRESSION, 5),
    )


def load_sensitivity_data():
    csv_files = sorted(glob("../data/test/*_sensitivity/stats.csv"))
    frames = []
    for csv_path in csv_files:
        print ("reading", csv_path)
        sensitivity_dir = os.path.basename(os.path.dirname(csv_path))
        center_name = sensitivity_dir.replace("_sensitivity", "")
        df = pd.read_csv(csv_path)
        df["center_name"] = center_name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def build_center_labels(df):
    unique_centers = list(dict.fromkeys(df["center_name"]))
    label_map = {raw: f"Site {['I','II','III','IV','V','VI'][i]}" for i, raw in enumerate(unique_centers)}
    df["center"] = df["center_name"].map(label_map)
    categories = [label_map[c] for c in unique_centers]
    df["center"] = pd.Categorical(df["center"], categories=categories, ordered=True)
    return df, categories


def compute_diffs(df, level_col):
    sub = df.dropna(subset=[level_col]).copy()
    sub["_case_key"] = sub["path"].apply(lambda p: os.path.basename(p).rsplit(f"_{level_col}_", 1)[0])
    result_rows = []
    for case_key, group in sub.groupby("_case_key", sort=False):
        baseline_rows = group[group[level_col] == 0]
        if len(baseline_rows) != 1:
            print(f"  WARNING: no baseline (level=0) found for case '{case_key}', skipping")
            raise Exception ("No/Too many baseline!", case_key)
        baseline = baseline_rows.iloc[0]

        for _, row in group[group[level_col] > 0].iterrows():
            new_row = row.copy()
            new_row["iou_tra_diff_abs"] = abs(row["iou_tra"]        - baseline["iou_tra"])
            new_row["iou_cor_diff_abs"] = abs(row["iou_cor"]        - baseline["iou_cor"])
            new_row["angle_diff_abs"]   = abs(row["angle_diff_cor"] - baseline["angle_diff_cor"])
            result_rows.append(new_row)

    result = pd.DataFrame(result_rows).drop(columns=["_case_key"])
    result = result.reset_index(drop=True)
    return result


def make_palette(n_centers: int):
    base = ["salmon", "limegreen", "slateblue", "darkorange", "mediumpurple", "teal"]
    return base[:n_centers]


def plot_diffs(data, ax, x_col, y_col, title, xlabel, ylabel,
               tick_values, tick_labels, n_centers):
    colors = make_palette(n_centers)
    sns.set_palette(sns.color_palette(colors))
    sns.stripplot(x=x_col, y=y_col, hue="center", data=data, ax=ax, jitter=0.25)

    # map original numeric values -> x-axis positions 0,1,2,...
    pos_map = {v: i for i, v in enumerate(tick_values)}
    mean_values = (
        data.groupby(x_col)[y_col]
        .mean()
        .reset_index()
    )
    mean_values["x_pos"] = mean_values[x_col].map(pos_map)
    ax.plot(
        mean_values["x_pos"].values,
        mean_values[y_col].values,
        marker="o", color="red", linestyle="-",
    )

    ax.invert_yaxis()
    ax.set_title(title, fontsize=22)
    ax.set_xlabel(xlabel, fontsize=23)
    ax.set_ylabel(ylabel, fontsize=23)
    ax.tick_params(axis="x", which="both", labelsize=21)
    ax.tick_params(axis="y", which="both", labelsize=21)
    ax.legend(fontsize=21)
    ax.set_xticks(list(range(len(tick_values))))
    ax.set_xticklabels(tick_labels)


def render_panel(data, level_col, tick_values, tick_labels,
                 panel_labels, xlabel, n_centers, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(30, 7))
    metrics = [
        ("iou_tra_diff_abs", "Difference in |IoU (axial)|"),
        ("iou_cor_diff_abs", "Difference in |IoU (coronal)|"),
        ("angle_diff_cor_abs",    "Difference in absolute angle"),
    ]

    for ax, (y_col, ylabel), panel_label in zip(axes, metrics, panel_labels):
        plot_diffs(
            data, ax, level_col, y_col,
            title="", xlabel=xlabel, ylabel=ylabel,
            tick_values=tick_values, tick_labels=tick_labels,
            n_centers=n_centers,
        )
        ax.text(
            -0.15, -0.11, panel_label,
            transform=ax.transAxes,
            fontsize=27, fontweight="bold", va="bottom", ha="right",
        )

    plt.subplots_adjust(wspace=0.37)
    plt.savefig(out_path, dpi=250, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    os.makedirs("../results", exist_ok=True)

    df = load_sensitivity_data()
    df, center_categories = build_center_labels(df)
    n_centers = len(center_categories)
    print(f"Found {n_centers} center(s): {center_categories}")

    print("\nProcessing noise sensitivity …")
    noise_tick_values = [1, 2, 4, 8, 16]
    noise_tick_labels = [str(v) for v in noise_tick_values]

    noise_data = compute_diffs(df, "noise")

    render_panel(
        noise_data,
        level_col="noise",
        tick_values=noise_tick_values,
        tick_labels=noise_tick_labels,
        panel_labels=["A", "B", "C"],
        xlabel="Noise Level",
        n_centers=n_centers,
        out_path="./sensitivity_noise_plot.tiff",
    )

    print("Processing contrast sensitivity …")
    # contrast levels [1,2,4,8,16] were stored as integers;
    # display as the actual enhancement factor: 1 + cl/16
    contrast_tick_values = [1, 2, 4, 8, 16]
    contrast_tick_labels = [f"{1 + v/16:.4g}" for v in contrast_tick_values]

    contrast_data = compute_diffs(df, "contrast")
    render_panel(
        contrast_data,
        level_col="contrast",
        tick_values=contrast_tick_values,
        tick_labels=contrast_tick_labels,
        panel_labels=["D", "E", "F"],
        xlabel="Contrast Level",
        n_centers=n_centers,
        out_path="./sensitivity_contrast_plot.tiff",
    )

    print("\nMerging panels …")
    join_plots(".")

#
