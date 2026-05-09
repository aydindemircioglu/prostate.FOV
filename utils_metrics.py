import os
import numpy as np
import cv2
from glob import glob
import pandas as pd
import math
import argparse
from scipy.spatial.distance import directed_hausdorff



def renderRect (rect1):
    # fixes size canvas of sie 2048x2048
    yMax, xMax = 2048, 2048
    rectC = [k+256 for k in rect1]
    if np.min(rectC) < 0:
        raise Exception("Not enough in - dir")
    if np.max(rectC) > 2047:
        raise Exception("Not enough in + dir")
    displayImgA = np.zeros((yMax, xMax, 1), dtype = np.uint8)
    lineThickness = 1
    _ = cv2.line(displayImgA, (int(rectC[0]), int(rectC[1])), (int(rectC[2]), int(rectC[3])), color = 255, thickness = lineThickness)
    _ = cv2.line(displayImgA, (int(rectC[2]), int(rectC[3])), (int(rectC[4]), int(rectC[5])), color = 255, thickness = lineThickness)
    _ = cv2.line(displayImgA, (int(rectC[4]), int(rectC[5])), (int(rectC[6]), int(rectC[7])), color = 255, thickness = lineThickness)
    _ = cv2.line(displayImgA, (int(rectC[6]), int(rectC[7])), (int(rectC[0]), int(rectC[1])), color = 255, thickness = lineThickness)
    seed_point = ( int(np.mean(rectC[::2][:4])), int(np.mean(rectC[1::2][:4]) ))
    _ = cv2.floodFill(displayImgA, None, seedPoint=seed_point, newVal=255)
    return displayImgA



def dice(im1, im2):
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def iou (imgA, imgB):
    # expect 0..255
    if imgA.shape != imgB.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    imgA = imgA > 0
    imgB = imgB > 0
    intersection = np.sum(np.logical_and(imgA, imgB))
    union = np.sum(np.logical_or(imgA, imgB))
    #print (intersection, union)

    iou = intersection/union
    return iou
# def iou(im1, im2):
#     im1 = np.asarray(im1).astype(bool)
#     im2 = np.asarray(im2).astype(bool)
#     intersection = np.logical_and(im1, im2).sum()
#     union = np.logical_or(im1, im2).sum()
#     return intersection / union if union > 0 else 0.0




#
# def reorder_rectangles(rectangles):
#     reordered_rectangles = []
#     for rectangle in rectangles:
#         # Unpack rectangle coordinates
#         x0, y0, x1, y1, x2, y2, x3, y3, conf = rectangle
#         # Calculate the midpoint
#         mid_x = (x0 + x1 + x2 + x3) / 4
#         mid_y = (y0 + y1 + y2 + y3) / 4
#         # Create a list to store the points and their relative positions
#         points = [(x, y) for x, y in [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]]
#         # Sort the points based on their relative positions to the midpoint
#         sorted_points = sorted(points, key=lambda point: (point[1] > mid_y, -point[0] if point[1] <= mid_y else point[0]))
#         # Reorder the points
#         reordered_rectangle = sum(sorted_points, ())
#         z = list(reordered_rectangle)
#         z.append(conf)
#         reordered_rectangles.append(z)
#     return reordered_rectangles
#


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



def normalize_angle(angle):
    angle += 180 if angle < 0 else 0
    angle %= 90
    if angle > 45:
        angle -= 90
    return angle



def computeAlpha(original_coordinates, perturbed_coordinates, verbose = False):
    original_coordinates = [(original_coordinates[i], original_coordinates[i+1]) for i in range(0, len(original_coordinates)-1, 2)]
    perturbed_coordinates = [(perturbed_coordinates[i], perturbed_coordinates[i+1]) for i in range(0, len(perturbed_coordinates)-1, 2)]

    original_angles = [ calculate_angle(original_coordinates[i], original_coordinates[(i + 1) % 4]) for i in range(4)]
    perturbed_angles = [calculate_angle(perturbed_coordinates[i], perturbed_coordinates[(i + 1) % 4]) for i in range(4)]
    original_angles = [normalize_angle(angle) for angle in original_angles]
    perturbed_angles = [normalize_angle(angle) for angle in perturbed_angles]
    if verbose == True:
        print ("original_angles\t\t\t", original_angles)
        print ("perturbed_angles\t\t\t", perturbed_angles)

    angle_differences = [original_angles[i] - perturbed_angles[i] for i in range(4)]
    # this fix needed for 2061040, but will be detected by raters anyway
    if np.mean(angle_differences) > 45:
        angle_differences = [a-45 for a in angle_differences]
    if np.mean(angle_differences) < -45:
        angle_differences = [a+45 for a in angle_differences]

    # Calculate the average angle difference
    if verbose == True:
        print ("DIFFS\t\t\t", angle_differences)
    average_angle_diff = np.mean(angle_differences)
    return average_angle_diff
