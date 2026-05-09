import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind_from_stats
from scipy import stats
from glob import glob


worst_TEST_CONFIGS = {
    "slice_diff":     dict(ref_mean=0.61,   ref_std=0.79,  delta=0.20,  alternative="less"),
    "angle_diff_abs": dict(ref_mean=5.09,   ref_std=4.81,   delta=0.05, alternative="less"),
    "iou_cor":        dict(ref_mean=0.7196, ref_std=0.0718, delta=0.15, alternative="greater"),
    "iou_tra":        dict(ref_mean=0.697,  ref_std=0.0696, delta=0.15, alternative="greater"),
    'center_dist_cor_perc':        dict(ref_mean=6.42,  ref_std=2.07, delta=2.5, alternative="less"),
    'center_dist_tra_perc':        dict(ref_mean=11.2,  ref_std=3.69, delta=2.5, alternative="less"),
    'hd_cor_perc':        dict(ref_mean=11.52,  ref_std=4.09, delta=2.5, alternative="less"),
    'hd_tra_perc':        dict(ref_mean=15.27,  ref_std=4.48, delta=2.5, alternative="less"),
}


TEST_CONFIGS = worst_TEST_CONFIGS

def _pval(cfg, net_mean, net_std, n):
    sign = -1 if cfg["alternative"] == "greater" else 1   # ← was +1/−1, now −1/+1
    boundary = cfg["ref_mean"] + sign * cfg["delta"]
    _, p = ttest_ind_from_stats(
        net_mean, net_std, n,               # ← net first
        boundary, cfg["ref_std"], n,
        equal_var=False,
        alternative=cfg["alternative"],
    )
    return p


def x_pval(cfg, net_mean, net_std, n):
    # boundary: the non-inferiority threshold
    # "greater" (IoU):  ref_mean - delta  → net must stay above this
    # "less" (errors):  ref_mean + delta  → net must stay below this
    sign = -1 if cfg["alternative"] == "greater" else 1
    boundary = cfg["ref_mean"] + sign * cfg["delta"]

    # one-sample t-test of net against fixed boundary
    t_stat = (net_mean - boundary) / (net_std / np.sqrt(n))
    t_stat = (net_mean - boundary) / (net_std)
    df = n - 1

    if cfg["alternative"] == "greater":
        p = stats.t.sf(t_stat, df)   # P(T > t_stat)
    else:
        p = stats.t.cdf(t_stat, df)  # P(T < t_stat)

    return p


def summarize_stats(stats_paths, metrics):
    dfs = []
    for p in stats_paths:
        df = pd.read_csv(p)
        if "true_slice" not in df.keys():
            continue # no GT
        df["center"] = Path(p).parent.name
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    if "predicted_slice" in data.columns and "true_slice" in data.columns:
        data["slice_diff"] = np.abs(data["predicted_slice"] - data["true_slice"])

    centers = data["center"].unique()
    center_label = {c:c for c in centers}
    for metric in metrics:
        if metric not in data.columns:
            print(f"  [missing column: {metric}]")
            continue
        is_iou = metric.startswith("iou")
        unit = "%" if is_iou else ""
        cfg = TEST_CONFIGS.get(metric)
        print(f"\n{'─'*50}\n{metric}")
        for center in centers:
            vals = data.loc[data["center"] == center, metric].dropna().astype(float)
            vals = vals[np.isfinite(vals)]
            disp = vals * 100 if is_iou else vals
            p_str = f"  p={_pval(cfg, np.mean(vals), np.std(vals), len(vals)):.4f}" if cfg else ""
            print(f"  {center_label.get(center, center):10s}  {np.mean(disp):+.2f}{unit} ± {np.std(disp, ddof=1):.2f}{unit}  (n={len(vals)}){p_str}")
        vals_all = data[metric].dropna().astype(float)
        vals_all = vals_all[np.isfinite(vals_all)]
        disp_all = vals_all * 100 if is_iou else vals_all
        p_str = f"  p={_pval(cfg, np.mean(vals_all), np.std(vals_all), len(vals_all)):.4f}" if cfg else ""
        print(f"  {'Pooled':10s}  {np.mean(disp_all):+.2f}{unit} ± {np.std(disp_all, ddof=1):.2f}{unit}  (n={len(vals_all)}){p_str}")

if __name__ == "__main__":
    centers = glob("../data/test/*/")
    stats_paths = [f"{s}/stats.csv" for s in centers]
    metrics = [
        'slice_diff',
        'iou_cor',
        'iou_tra',
        'angle_diff_cor_abs', # tra same, we only use one
        'center_dist_cor_perc',
        'center_dist_tra_perc',
        'hd_cor_perc',
        'hd_tra_perc',
    ]
    summarize_stats(stats_paths, metrics)
