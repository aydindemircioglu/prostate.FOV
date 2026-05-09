#!/usr/bin/python3
import os
import numpy as np
import cv2
from glob import glob
import pandas as pd
import math

import sys
sys.path.append("..")
from utils_compute import *
from utils_metrics import *

array = np.array

if __name__ == "__main__":
    centers = glob("../data/test/*/")
    for c in centers:
        print ("\n####### Processing center", c)
        try:
            stats = pd.read_csv(os.path.join(c,"stats.csv"))
        except:
            print ("Cannot find the center stats.csv")
            continue

        # do we have a ground truth? if not we cannot compute on that center
        try:
            for k in ["cor_LL_x", "cor_LL_y", "cor_LR_x", "cor_LR_y", "cor_UR_x","cor_UR_y", "cor_UL_x",  "cor_UL_y"]:
                z = stats[k]
        except Exception as e:
            print ("Center has no GT!")
            continue

        print (stats.keys())
        for i, (idx, row) in enumerate(stats.iterrows()):
            try:
                # extract them from row befre we temporarily overwrite them
                gt_cor = [row[k] for k in ["cor_LL_x", "cor_LL_y", "cor_LR_x", "cor_LR_y", "cor_UR_x","cor_UR_y", "cor_UL_x",  "cor_UL_y"]]
                gt_tra = [row[k] for k in ["tra_LL_x", "tra_LL_y", "tra_LR_x", "tra_LR_y", "tra_UR_x","tra_UR_y", "tra_UL_x",  "tra_UL_y"]]
            except Exception as e:
                raise Exception ("Should never happen?")
                continue

            cor = eval(row["stage2_cor"])
            tra = eval(row["stage2_tra"])

            # centers
            c_pred = (int(np.mean(cor[0::2][:4])), int(np.mean(cor[1::2][:4])))
            c_gt = (int(np.mean(gt_cor[0::2][:4])), int(np.mean(gt_cor[1::2][:4])))

            overlap_cor = iou (renderRect(cor), renderRect(gt_cor))
            overlap_tra = iou (renderRect(tra), renderRect(gt_tra))
            stats.at[idx, "iou_cor"] = overlap_cor
            stats.at[idx, "iou_tra"] = overlap_tra

            alpha_cor = computeAlpha (cor, gt_cor)
            alpha_tra = computeAlpha (tra, gt_tra)
            # also compute dice and angle diff, cor=tra, because we fixed the cor rectangle to be ortho
            stats.at[idx, "angle_diff_cor"] = alpha_cor
            stats.at[idx, "angle_diff_cor_abs"] = np.abs(alpha_cor)
            stats.at[idx, "angle_diff_tra"] = alpha_tra
            stats.at[idx, "angle_diff_tra_abs"] = np.abs(alpha_tra)

            dist_px, dist_mm, dist_perc = compute_center_dist(cor, gt_cor, row['PX'], row['PY'])
            stats.at[idx, "center_dist_cor_mm"] = dist_mm
            stats.at[idx, "center_dist_cor_perc"] = dist_perc
            dist_px, dist_mm, dist_perc = compute_center_dist(tra, gt_tra, row['PX'], row['PY'])
            stats.at[idx, "center_dist_tra_mm"] = dist_mm
            stats.at[idx, "center_dist_tra_perc"] = dist_perc

            hd_cor = compute_hausdorff(cor, gt_cor, row['PX'], row['PY'], hd95 = True)
            stats.at[idx, "hd_cor_mm"] = hd_cor
            hd_tra = compute_hausdorff(tra, gt_tra, row['PX'], row['PY'], hd95 = True)
            stats.at[idx, "hd_tra_mm"] = hd_tra

            hd_cor_perc = compute_hausdorff(cor, gt_cor, row['PX'], row['PY'], perc = True, hd95 = True)
            stats.at[idx, "hd_cor_perc"] = hd_cor_perc
            hd_tra_perc = compute_hausdorff(tra, gt_tra, row['PX'], row['PY'], perc = True, hd95 = True)
            stats.at[idx, "hd_tra_perc"] = hd_tra_perc

        # save
        stats.to_csv(f'{c}/stats.csv', index=False)
        print(f'saving to {c}/stats.csv')

        # report
        # angle should be same
        print ("Angle diff cor mean", np.nanmean(np.abs(stats["angle_diff_cor"])))
        print ("Angle diff cor STD", np.nanstd(np.abs(stats["angle_diff_cor"])))
        print ("Angle diff tra mean", np.nanmean(np.abs(stats["angle_diff_tra"])))
        print ("Angle diff tra STD", np.nanstd(np.abs(stats["angle_diff_tra"])))

        print("Center dist Cor mean (mm)", np.nanmean(stats["center_dist_cor_mm"]))
        print("Center dist Cor STD (mm)", np.nanstd(stats["center_dist_cor_mm"]))
        print("Sorted Coronal Distances (mm):")
        # print(stats[["path", "center_dist_cor_mm"]].sort_values("center_dist_cor_mm").to_string(index=False))

        print("Center dist Cor mean (perc)", np.nanmean(stats["center_dist_cor_perc"]))
        print("Center dist Cor STD (perc)", np.nanstd(stats["center_dist_cor_perc"]))
        print("Sorted Coronal Distances (perc):")

        print("Hausdorff Cor mean (mm)", np.nanmean(stats["hd_cor_mm"]))
        print("Hausdorff Tra mean (mm)", np.nanmean(stats["hd_tra_mm"]))
        print("Sorted Coronal Hausdorff (mm):")

        print("Hausdorff Cor mean (%)", np.nanmean(stats["hd_cor_perc"]))
        print("Hausdorff Tra mean (%)", np.nanmean(stats["hd_tra_perc"]))
        print("Sorted Coronal Hausdorff (%):")

        print ("IoU diff mean", np.nanmean(np.abs(stats["iou_cor"])))
        print ("IoU diff STD", np.nanstd(np.abs(stats["iou_tra"])))

        print ("Slice diff mean", np.nanmean(np.abs(stats["true_slice"]-stats["predicted_slice"])))

        print ("Angle diff cor mean", np.nanmean(np.abs(stats["angle_diff_cor"])))
        print ("Angle diff cor median", np.nanmedian(np.abs(stats["angle_diff_cor"])))
        print ("Angle diff cor STD", np.nanstd(np.abs(stats["angle_diff_cor"])))
        print ("Angle diff tra mean", np.nanmean(np.abs(stats["angle_diff_tra"])))
        print ("Angle diff tra median", np.nanmedian(np.abs(stats["angle_diff_tra"])))
        print ("Angle diff tra STD", np.nanstd(np.abs(stats["angle_diff_tra"])))


#
