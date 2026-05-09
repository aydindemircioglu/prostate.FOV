#!/usr/bin/python3
import os
import re
import numpy as np
import pandas as pd
from glob import glob
import math
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.stats import ttest_ind, wilcoxon, ttest_rel
from sklearn.metrics import cohen_kappa_score
from scipy.spatial.distance import directed_hausdorff
from shapely.geometry import Polygon

import sys
sys.path.append("..")
from utils_compute import *
from utils_metrics import *


def iou(coords1, coords2):
    # coords: [LL_x, LL_y, LR_x, LR_y, UR_x, UR_y, UL_x, UL_y]
    def to_poly(c):
        pts = [(c[i], c[i+1]) for i in range(0, len(c), 2)]
        return Polygon(pts)

    p1 = to_poly(coords1)
    p2 = to_poly(coords2)

    intersection = p1.intersection(p2).area
    union = p1.union(p2).area
    return intersection / union if union > 0 else 0.0


def convertCoords(row, rect):
    r = [row[k] for k in [f"{rect}_LL_x", f"{rect}_LL_y", f"{rect}_LR_x", f"{rect}_LR_y",
                           f"{rect}_UR_x", f"{rect}_UR_y", f"{rect}_UL_x", f"{rect}_UL_y"]]
    return r


def getStats_Stage(A, B, s, returnDF=False):
    data = []
    for r1 in glob(f"./rater_{A}_{s}/*.csv"):
        r2 = f"./rater_{B}_{s}/{os.path.basename(r1)}"
        if not os.path.exists(r2):
            continue

        ann1 = pd.read_csv(r1).query("curFrame >= 0").reset_index(drop = True)
        ann2 = pd.read_csv(r2).query("curFrame >= 0").reset_index(drop = True)
        assert len(ann1) == 1
        assert len(ann2) == 1

        # pixel spacing is written into the annotation CSV by the annotation tool
        px = ann1.iloc[0]["pixelSpacingX"]
        py = ann1.iloc[0]["pixelSpacingY"]

        if s == 1:
            row = {f"Slice_diff_{A}_{B}": np.abs(ann1.iloc[0]["curFrame"] - ann2.iloc[0]["curFrame"])}
        else:
            ann1_tra = convertCoords(ann1.iloc[0], "tra")
            ann1_cor = convertCoords(ann1.iloc[0], "cor")
            ann2_tra = convertCoords(ann2.iloc[0], "tra")
            ann2_cor = convertCoords(ann2.iloc[0], "cor")

            overlap_cor = iou(ann1_cor, ann2_cor) * 100
            overlap_tra = iou(ann1_tra, ann2_tra) * 100
            alpha_cor = computeAlpha(ann1_cor, ann2_cor)
            alpha_tra = computeAlpha(ann1_tra, ann2_tra)

            # _, _, cd_cor_perc = compute_center_dist(ann1_cor, ann2_cor, px, py)
            # _, _, cd_tra_perc = compute_center_dist(ann1_tra, ann2_tra, px, py)
            # hd_cor_perc = compute_hausdorff(ann1_cor, ann2_cor, px, py, perc=True, hd95=True)
            # hd_tra_perc = compute_hausdorff(ann1_tra, ann2_tra, px, py, perc=True, hd95=True)

            row = {
                f"IoU_cor_{A}_{B}": overlap_cor,
                f"IoU_tra_{A}_{B}": overlap_tra,
                f"Alpha_cor_{A}_{B}": alpha_cor,
                f"Alpha_tra_{A}_{B}": alpha_tra,
                # f"CD_cor_perc_{A}_{B}": cd_cor_perc,
                # f"CD_tra_perc_{A}_{B}": cd_tra_perc,
                # f"HD_cor_perc_{A}_{B}": hd_cor_perc,
                # f"HD_tra_perc_{A}_{B}": hd_tra_perc,
            }
        data.append(row)

    if not data:
        return pd.Series(dtype=float)

    df = pd.DataFrame(data).reset_index(drop=True)
    dfA = np.abs(df).mean(axis=0)
    dfB = np.abs(df).std(axis=0)
    dfA.index = [z + "_mean" for z in dfA.index]
    dfB.index = [z + "_std" for z in dfB.index]
    result = pd.concat([dfA.to_frame().T, dfB.to_frame().T], axis=1).iloc[0]
    if returnDF:
        return result, df
    return result


def round_dict_values(d):
    return {k: round(v, 2) if isinstance(v, float) else v for k, v in d.items()}


def detect_raters(stages=(1, 2)):
    found = set()
    for stage in stages:
        for d in glob(f"./rater_*_{stage}"):
            if os.path.isdir(d):
                match = re.match(rf"^\./rater_(.+)_{stage}$", d)
                if match:
                    found.add(match.group(1))

    all_raters = sorted(found)

    base_map = {}
    for rater in all_raters:
        match = re.match(r"^(.+)_([a-z])$", rater)
        if match:
            base = match.group(1)
            base_map.setdefault(base, []).append(rater)
        else:
            base_map.setdefault(rater, []).append(rater)

    intra_groups = {base: members for base, members in base_map.items() if len(members) > 1}
    return all_raters, intra_groups


if __name__ == "__main__":
    stats = {}

    all_raters, intra_groups = detect_raters(stages=[1, 2])
    print("Detected raters:", all_raters)
    print("Intra-rater groups:", intra_groups)

    # intra-rater comparisons
    for base, members in intra_groups.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                r1, r2 = members[i], members[j]
                for s in [1, 2]:
                    result = getStats_Stage(r1, r2, s)
                    if not result.empty:
                        stats.update(result.to_dict())

    # inter-rater comparisons
    for i, r1 in enumerate(all_raters):
        for j, r2 in enumerate(all_raters):
            if i < j:
                for s in [1, 2]:
                    result = getStats_Stage(r1, r2, s)
                    if not result.empty:
                        stats.update(result.to_dict())

    stats = round_dict_values(stats)
    pprint(stats)

#
