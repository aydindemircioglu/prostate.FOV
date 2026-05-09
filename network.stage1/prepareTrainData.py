#!/usr/bin/python3
import os
import random
from sklearn.model_selection import KFold
from joblib import dump, load
import SimpleITK as sitk
import pandas as pd
import numpy as np
from skimage.draw import line_aa
import cv2
from scipy import ndimage
import sys
from pprint import pprint
from PIL import Image

from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import shutil
from random import sample
import imageio
import pydicom

sys.path.append("../")
from annotate import loadMR
from utils_montage import *
from utils_path import recreatePath
from utils_annotation import *


exportPath = "../data/stage1/"



def exportMontageAnnotation (patFrames, exportPath, subdir, doPaint = False):
    os.makedirs (f"{exportPath}/{subdir}", exist_ok = True)
    os.makedirs (f"{exportPath}/{subdir}/images", exist_ok = True)
    os.makedirs (f"{exportPath}/{subdir}/annotations", exist_ok = True)
    assert(len(set(patFrames["path"].values)) == 1)

    # load MR from first annotation
    ann = patFrames.iloc[0:1].reset_index().copy()
    ann.at[0,"path"] = os.path.join("../", ann.iloc[0]["path"])
    try:
        mrImg, zsp = loadMR(ann)
    except:
        print ("unable to load MR:", imageName)
        return None

    # create montage with all
    baseName = '_'.join(ann.iloc[0]["path"].split("/")[-3:])
    print (baseName)

    # for testing ensure that we did not add random black frames to montage
    frame_offset = None
    if "test" in subdir:
        frame_offset = 0
    montage, bboxes = getMontage(mrImg, patFrames, frame_offset = frame_offset)

    # save both, can reuse ann
    imagePath = f"{exportPath}/{subdir}/images/{baseName}.png"
    annPath = f"{exportPath}/{subdir}/annotations/{baseName}.txt"
    cv2.imwrite(imagePath, montage)

    # first write "true"
    with open(annPath, "w") as text_file:
        text_file.write(bboxes)

    if doPaint == True:
        renderOverlay (imagePath = imagePath, annPath = annPath)



def renderOverlay (imagePath, annPath):
    img = cv2.imread(imagePath)
    ann = pd.read_csv(annPath, sep = " ", header = None)
    colormap = {"box": (0,255,0)}

    # check all lines
    for k in range(len(ann)):
        cv2.line(img, (int(ann.iloc[k][0]), int(ann.iloc[k][1])), (int(ann.iloc[k][2]), int(ann.iloc[k][3])), color = colormap[ann.iloc[k][8]], thickness = 2)
        cv2.line(img, (int(ann.iloc[k][2]), int(ann.iloc[k][3])), (int(ann.iloc[k][4]), int(ann.iloc[k][5])), color = colormap[ann.iloc[k][8]], thickness = 2)
        cv2.line(img, (int(ann.iloc[k][4]), int(ann.iloc[k][5])), (int(ann.iloc[k][6]), int(ann.iloc[k][7])), color = colormap[ann.iloc[k][8]], thickness = 2)
        cv2.line(img, (int(ann.iloc[k][6]), int(ann.iloc[k][7])), (int(ann.iloc[k][0]), int(ann.iloc[k][1])), color = colormap[ann.iloc[k][8]], thickness = 2)

    # save it to tmp
    cv2.imwrite(f"./tmp/{os.path.basename(imagePath)}", img)
    return None



if __name__ == "__main__":
    recreatePath (exportPath)
    trainpath = os.path.join(exportPath, "train")
    recreatePath (trainpath)

    allAnn, allAcc = getAllValidAnnotations()

    # now training files
    nCV = 5
    kf = KFold(n_splits = nCV, shuffle = True, random_state = 42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(allAcc)):
        tmp = allAcc[train_idx]
        for pID in tmp:
            patFrames = allAnn.query("AccNr == @pID").copy()
            exportMontageAnnotation(patFrames, exportPath, f"fold_{fold}/train")

        tmp = allAcc[test_idx]
        for pID in tmp:
            patFrames = allAnn.query("AccNr == @pID").copy()
            exportMontageAnnotation(patFrames, exportPath, f"fold_{fold}/test")

#
