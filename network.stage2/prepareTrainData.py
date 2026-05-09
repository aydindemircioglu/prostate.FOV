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
from utils_montage import pseudoRGB
from utils_path import recreatePath

exportPath = "../data/stage2/"




def exportAnnotation (patFrames, exportPath, subdir):
    os.makedirs (f"{exportPath}/{subdir}", exist_ok = True)
    os.makedirs (f"{exportPath}/{subdir}/images", exist_ok = True)
    os.makedirs (f"{exportPath}/{subdir}/annotations", exist_ok = True)

    # load MR from first annotation
    ann = patFrames.iloc[0:1].reset_index().copy()
    ann.at[0,"path"] = os.path.join("../", ann.iloc[0]["path"])
    try:
        mrImg, zsp = loadMR(ann)
    except:
        print ("unable to load MR:", imageName)
        return None

    # create montage with all
    for j in range(len(patFrames)):
        ann = patFrames.iloc[j]
        z = ann["curFrame"]
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = getBBox(ann, "cor")
        corrow = "{} {} {} {} {} {} {} {} cor 0".format(x0, y0, x1, y1, x2, y2, x3, y3)#, int(pgon.area))
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = getBBox(ann, "tra")
        trarow = "{} {} {} {} {} {} {} {} tra 0".format(x0, y0, x1, y1, x2, y2, x3, y3)#, int(pgon.area))
        if x0 == -1:
            print (corrow)
            print (trarow)
            print ("DROP")
            continue

        mrSlice  = mrImg[z,:,:]
        # slice-wise normalization in addition
        neu = (mrSlice-np.min(mrSlice))/(np.max(mrSlice)-np.min(mrSlice)) * 255.0
        mrSlice = np.asarray(neu, dtype = np.uint8)
        image = np.stack((mrSlice[:,:],)*3, axis = -1)
        imageRGB = pseudoRGB (image[:,:,0])

        # save both
        baseName = '_'.join(ann["path"].split("/")[-3:])
        imagePath = f"{exportPath}/{subdir}/images/{baseName}_{z}.png"
        annPath = f"{exportPath}/{subdir}/annotations/{baseName}_{z}.txt"
        cv2.imwrite(imagePath, imageRGB)

        # first write "true"
        with open(annPath, "w") as text_file:
           text_file.write(corrow + "\n")
           text_file.write(trarow + "\n")

        renderOverlay (imagePath = imagePath, annPath = annPath)



def renderOverlay (imagePath, annPath):
    os.makedirs("./tmp", exist_ok = True)
    img = cv2.imread(imagePath)
    ann = pd.read_csv(annPath, sep = " ", header = None)
    colormap = {"tra": (255,0,255), "cor":(255,0,0)}
    # know we have two lines
    cv2.line(img, (int(ann.iloc[0][0]), int(ann.iloc[0][1])), (int(ann.iloc[0][2]), int(ann.iloc[0][3])), color = colormap[ann.iloc[0][8]], thickness = 2)
    cv2.line(img, (int(ann.iloc[0][2]), int(ann.iloc[0][3])), (int(ann.iloc[0][4]), int(ann.iloc[0][5])), color = colormap[ann.iloc[0][8]], thickness = 2)
    cv2.line(img, (int(ann.iloc[0][4]), int(ann.iloc[0][5])), (int(ann.iloc[0][6]), int(ann.iloc[0][7])), color = colormap[ann.iloc[0][8]], thickness = 2)
    cv2.line(img, (int(ann.iloc[0][6]), int(ann.iloc[0][7])), (int(ann.iloc[0][0]), int(ann.iloc[0][1])), color = colormap[ann.iloc[0][8]], thickness = 2)

    cv2.line(img, (int(ann.iloc[1][0]), int(ann.iloc[1][1])), (int(ann.iloc[1][2]), int(ann.iloc[1][3])), color = colormap[ann.iloc[1][8]], thickness = 2)
    cv2.line(img, (int(ann.iloc[1][2]), int(ann.iloc[1][3])), (int(ann.iloc[1][4]), int(ann.iloc[1][5])), color = colormap[ann.iloc[1][8]], thickness = 2)
    cv2.line(img, (int(ann.iloc[1][4]), int(ann.iloc[1][5])), (int(ann.iloc[1][6]), int(ann.iloc[1][7])), color = colormap[ann.iloc[1][8]], thickness = 2)
    cv2.line(img, (int(ann.iloc[1][6]), int(ann.iloc[1][7])), (int(ann.iloc[1][0]), int(ann.iloc[1][1])), color = colormap[ann.iloc[1][8]], thickness = 2)

    # save it to tmp
    cv2.imwrite(f"./tmp/{os.path.basename(imagePath)}", img)
    return None



def getBBox (a, oclass = "cor"):
    if oclass == "cor":
        x0, y0 = a["cor_LL_x"], a["cor_LL_y"]
        x1, y1 = a["cor_LR_x"], a["cor_LR_y"]
        x2, y2 = a["cor_UR_x"], a["cor_UR_y"]
        x3, y3 = a["cor_UL_x"], a["cor_UL_y"]
    elif oclass == "tra":
        x0, y0 = a["tra_LL_x"], a["tra_LL_y"]
        x1, y1 = a["tra_LR_x"], a["tra_LR_y"]
        x2, y2 = a["tra_UR_x"], a["tra_UR_y"]
        x3, y3 = a["tra_UL_x"], a["tra_UL_y"]
    else:
        raise Exception("unknown class.")

    return (x0, y0), (x1, y1), (x2, y2), (x3, y3)



def getAllAnnotations():
    aList = sorted(glob (f"../annotations/train/*.csv"))
    print ("Have", len(aList), "annotations")
    invList = []
    vList = []
    for fann in aList:
        ann = pd.read_csv (fann)
        ann = ann.query('curFrame > 0').copy()
        if len(ann) == 0:
            print(ann)
            continue
        vList.append(ann)
    print (f"Valid annotations: {len(vList)}")
    return vList





if __name__ == "__main__":
    recreatePath (exportPath)
    trainpath = os.path.join(exportPath, "train")
    recreatePath (trainpath)


    allAnn = getAllAnnotations()
    allAnn = pd.concat(allAnn).reset_index(drop = True)
    # we have _mid_ keys, these are not ok, not clear how they made it into the annotation
    allAnn = allAnn.drop([k for k in allAnn.keys() if "_mid" in k or "_ort" in k or "_par" in k or "Alpha" in k], axis = 1).copy()

    # we have no deleted, but nonetheless
    allAnn = allAnn.query("Deleted == 0").copy()

    # we split by patient, so we need this
    allAnn["AccNr"] = [k.split("/")[-2] for k in allAnn["path"]]
    allAcc = np.array(sorted(list(set(allAnn["AccNr"].values))))



    # now training files
    nCV = 5
    kf = KFold(n_splits = nCV, shuffle = True, random_state = 42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(allAcc)):
        tmp = allAcc[train_idx]
        for pID in tmp:
            patFrames = allAnn.query("AccNr == @pID").copy()
            exportAnnotation(patFrames, exportPath, f"fold_{fold}/train")

        tmp = allAcc[test_idx]
        for pID in tmp:
            patFrames = allAnn.query("AccNr == @pID").copy()
            exportAnnotation(patFrames, exportPath, f"fold_{fold}/test")


#
