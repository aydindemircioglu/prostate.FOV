#!/usr/bin/python3
import os
import numpy as np
import cv2
from glob import glob
import pandas as pd
import math
import argparse
from scipy.spatial.distance import directed_hausdorff
import sys
sys.path.append("..")
from utils_compute import *
from utils_metrics import *


def drawAnnImg (oimg, curAnnotation, lineColor, lineThickness = 1):
    ofs = 0
    _ = cv2.line(oimg, (int(curAnnotation["cor_LL_x"])-ofs, int(curAnnotation["cor_LL_y"])-ofs), (int(curAnnotation["cor_LR_x"])-ofs, int(curAnnotation["cor_LR_y"])-ofs), color = lineColor, thickness = lineThickness)
    _ = cv2.line(oimg, (int(curAnnotation["cor_LL_x"])-ofs, int(curAnnotation["cor_LL_y"])-ofs), (int(curAnnotation["cor_UL_x"])-ofs, int(curAnnotation["cor_UL_y"])-ofs), color = lineColor, thickness = lineThickness)
    _ = cv2.line(oimg, (int(curAnnotation["cor_LR_x"])-ofs, int(curAnnotation["cor_LR_y"])-ofs), (int(curAnnotation["cor_UR_x"])-ofs, int(curAnnotation["cor_UR_y"])-ofs), color = lineColor, thickness = lineThickness)
    _ = cv2.line(oimg, (int(curAnnotation["cor_UR_x"])-ofs, int(curAnnotation["cor_UR_y"])-ofs), (int(curAnnotation["cor_UL_x"])-ofs, int(curAnnotation["cor_UL_y"])-ofs), color = lineColor, thickness = lineThickness)

    lineColor = [l-l//8 for l in lineColor]
    _ = cv2.line(oimg, (int(curAnnotation["tra_LL_x"])-ofs, int(curAnnotation["tra_LL_y"])-ofs), (int(curAnnotation["tra_LR_x"])-ofs, int(curAnnotation["tra_LR_y"])-ofs), color = lineColor, thickness = lineThickness)
    _ = cv2.line(oimg, (int(curAnnotation["tra_LL_x"])-ofs, int(curAnnotation["tra_LL_y"])-ofs), (int(curAnnotation["tra_UL_x"])-ofs, int(curAnnotation["tra_UL_y"])-ofs), color = lineColor, thickness = lineThickness)
    _ = cv2.line(oimg, (int(curAnnotation["tra_LR_x"])-ofs, int(curAnnotation["tra_LR_y"])-ofs), (int(curAnnotation["tra_UR_x"])-ofs, int(curAnnotation["tra_UR_y"])-ofs), color = lineColor, thickness = lineThickness)
    _ = cv2.line(oimg, (int(curAnnotation["tra_UR_x"])-ofs, int(curAnnotation["tra_UR_y"])-ofs), (int(curAnnotation["tra_UL_x"])-ofs, int(curAnnotation["tra_UL_y"])-ofs), color = lineColor, thickness = lineThickness)
    return (oimg)


def find_middle_rectangle(rectangles):
    max_overlap = 0
    middle_rectangle = None

    if len(rectangles) == 1:
        return rectangles[0]

    for i, rect1 in enumerate(rectangles):
        overlap_area = 0
        for j, rect2 in enumerate(rectangles):
            if i != j:
                overlap_area += dice (renderRect(rect1), renderRect(rect2))

        if overlap_area > max_overlap:
            max_overlap = overlap_area
            middle_rectangle = rect1

    return middle_rectangle



def calculate_barycenter(coordinates):
    x_coords = [point[0] for point in coordinates]
    y_coords = [point[1] for point in coordinates]
    barycenter = (np.mean(x_coords), np.mean(y_coords))
    return barycenter


def calculate_angle(point1, point2):
    return np.degrees(np.arctan2(point2[1] - point1[1], point2[0] - point1[0]))


def fix_coordinates(original_coordinates, perturbed_coordinates, verbose=False):
    if verbose:
        print("ORIGINAL", original_coordinates)
        print("PERTURBED", perturbed_coordinates)

    # Calculate the angles of the sides of the original rectangle
    original_angles = [
        calculate_angle(original_coordinates[i], original_coordinates[(i + 1) % 4])
        for i in range(4)
    ]

    # Calculate the angles of the sides of the perturbed rectangle
    perturbed_angles = [
        calculate_angle(perturbed_coordinates[i], perturbed_coordinates[(i + 1) % 4])
        for i in range(4)
    ]

    # Calculate the angle difference between the corresponding sides
    angle_differences = [
        original_angles[i] - perturbed_angles[i]
        for i in range(4)
    ]

    # Calculate the barycenter of the perturbed_coordinates
    perturbed_barycenter = calculate_barycenter(perturbed_coordinates)

    # Calculate the average angle difference
    average_angle_diff = np.mean(angle_differences)

    if verbose:
        print("Original Angles", original_angles)
        print("Perturbed Angles", perturbed_angles)
        print("Angle Differences", angle_differences)
        print("Perturbed Barycenter", perturbed_barycenter)

    # If the rotation is larger than 45 degrees, reverse the rotation direction
    if abs(average_angle_diff) > 45:
        corrected_coordinates = [
            (
                np.cos(np.radians(-90 + average_angle_diff)) * (point[0] - perturbed_barycenter[0]) -
                np.sin(np.radians(-90 + average_angle_diff)) * (point[1] - perturbed_barycenter[1]) +
                perturbed_barycenter[0],
                np.sin(np.radians(-90 + average_angle_diff)) * (point[0] - perturbed_barycenter[0]) +
                np.cos(np.radians(-90 + average_angle_diff)) * (point[1] - perturbed_barycenter[1]) +
                perturbed_barycenter[1]
            )
            for point in perturbed_coordinates
        ]
    else:
        corrected_coordinates = [
            (
                np.cos(np.radians(average_angle_diff)) * (point[0] - perturbed_barycenter[0]) -
                np.sin(np.radians(average_angle_diff)) * (point[1] - perturbed_barycenter[1]) +
                perturbed_barycenter[0],
                np.sin(np.radians(average_angle_diff)) * (point[0] - perturbed_barycenter[0]) +
                np.cos(np.radians(average_angle_diff)) * (point[1] - perturbed_barycenter[1]) +
                perturbed_barycenter[1]
            )
            for point in perturbed_coordinates
        ]

    if verbose:
        print("CORRECTED", corrected_coordinates)

    return corrected_coordinates




if __name__ == "__main__":
    centers = glob("../data/test/*/")
    print ("Predicting on centers", centers)

    for c in centers:
        print ("\n####### Processing center", c)
        try:
            stats = pd.read_csv(os.path.join(c,"stats.csv"))
        except:
            print ("No stats, no prediction!")
            exit(-1)

        brokenstats = []
        # i hate you all
        array = np.array
        for i, (idx, row) in enumerate(stats.iterrows()):
            result_stage2 = {"cor": [], "tra":[]}
            for f in range(5):
                try:
                    cor = eval(row[f"stage2_{f}_cor"])
                    tra = eval(row[f"stage2_{f}_tra"])
                    # check (that happened) for a rectangle out-of-space
                    if np.max(cor) > 4096 or np.min(cor) < -4096 or np.max(tra) > 4096 or np.min(tra) < -4096:
                        print ("###### DISCARDED\n", cor, "\n", tra, "<<<<< ")
                        continue
                    result_stage2["cor"].append(cor)
                    result_stage2["tra"].append(tra)
                except:
                    # no pred
                    continue

            fSelectedSliceImage = os.path.join(row.path, "Selected_Slice_RGB.png")
            img = cv2.imread(fSelectedSliceImage)
            img[:,:,0] = img[:,:,1]
            img[:,:,2] = img[:,:,1]

            if len(result_stage2["cor"]) == 0 and len(result_stage2["tra"]) == 0:
                #cv2.imwrite (f"totally_broken_{c}_{i}.png", img)
                print ("No prediction at all.")
                stats.at[idx, f"stage2_prediction"] = 0
                brokenstats.append(row)
                continue
            stats.at[idx, f"stage2_prediction"] = 1


            def average_coordinates(rectangles):
                if not rectangles:
                    return None
                mean_rectangle = np.median(rectangles, axis=0)
                return mean_rectangle

            cor = find_middle_rectangle(result_stage2['cor'])
            tra = find_middle_rectangle(result_stage2['tra'])

            # now fix
            cvt_tra = [(tra[i], tra[i+1]) for i in range(0, len(tra)-1, 2)]
            cvt_cor = [(cor[i], cor[i+1]) for i in range(0, len(cor)-1, 2)]
            cvt_tra = fix_coordinates(cvt_cor, cvt_tra, False)
            tra = [item for tup in cvt_tra for item in tup] + [tra[-1]]

            # put back to stats as well
            stats.at[idx, f"stage2_cor"] = repr(cor)
            stats.at[idx, f"stage2_tra"] = repr(tra)

            row = row.copy() # safety first
            row["cor_LL_x"], row["cor_LR_x"], row["cor_UL_x"], row["cor_UR_x"] = cor[0], cor[2], cor[6], cor[4]
            row["cor_LL_y"], row["cor_LR_y"], row["cor_UL_y"], row["cor_UR_y"] = cor[1], cor[3], cor[7], cor[5]
            row["tra_LL_x"], row["tra_LR_x"], row["tra_UL_x"], row["tra_UR_x"] = tra[0], tra[2], tra[6], tra[4]
            row["tra_LL_y"], row["tra_LR_y"], row["tra_UL_y"], row["tra_UR_y"] = tra[1], tra[3], tra[7], tra[5]

            fFinalSlice = os.path.join(row.path, "Final_Slice_Prediction.png")
            img = drawAnnImg (img, row, (255,0,0))
            cv2.imwrite (fFinalSlice, img)


        stats = stats.loc[:, ~stats.columns.str.contains('Unnamed')]
        stats.to_csv(f'{c}/stats.csv', index=False)
        print(f'saving to {c}/stats.csv')

        df = pd.DataFrame(brokenstats).reset_index(drop = True)
        df = df.drop([f for f in df.keys() if "cor_" in f or "_cor" in f or "tra_" in f or "_tra" in f], axis = 1).copy()
        df.to_csv(os.path.join(c, "broken_stats_stage2.csv"), index=False)

#
