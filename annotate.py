#!/usr/bin/python3

import os
from joblib import dump, load
import SimpleITK as sitk
import pandas as pd
import numpy as np
from skimage.draw import line_aa
import cv2
from scipy import ndimage
import sys
from pprint import pprint
import math
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import shutil
from random import sample
import imageio
import pydicom
import argparse


font = cv2.FONT_HERSHEY_SIMPLEX

K_7 = 55
K_9 = 57
K_q = 113
K_e = 101
K_d = 100
K_a = 97
K_h = 104
K_o = 111
K_p = 112
K_z = K_q+9
K_x = K_q+7

K_comma = 44
K_semicolon = 46
K_f = 102
K_g = 103
K_h = 104
K_t = 116

K_i = 105
K_j = 106
K_k = 107
K_l = 108

K_m = 109
K_n = 110

DELETE = 8
SKIP = 32
BACK = 96  # MAC 167, ME 96
ESC = 27



clipFactor = 2
clipLimit = 2

wmin, wmax = None, None
basePath = "./data/"
bboxcolors = {"cor": (255, 0 ,0), "tra": (0, 255, 255)}

k = 0
wmb = 0
maxFrame = 0
mousePressed = 0


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--cohort', type=str, default='train', help='cohort to annotate.')
        self.parser.add_argument('-f', type = str, default = None, help = 'dummy.')

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        return self.opt



# mouse callback function
def mouseCallback(event,x,y,flags,param):
    global ix,iy,drawing,mode, curAnnotation, curFrame, maxFrame, mousePressed
    mousePressed  = 0
    if event == cv2.EVENT_LBUTTONDOWN:
        ix,iy = x,y
    elif event == cv2.EVENT_LBUTTONUP:
        curAnnotation["curFrame"] = curFrame
        curAnnotation["cor_mid_x"] = int((x+ix)/2)
        curAnnotation["cor_mid_y"] = int((y + iy)/2)
        curAnnotation["tra_mid_x"] = int((x+ix)/2)
        curAnnotation["tra_mid_y"] = int((y + iy)/2)

        if x-ix != 0:
            curAnnotation["Alpha"] =  np.arctan2((y-iy), (x-ix)) - np.pi/2
            #np.arctan2(1,4)
        else:
            curAnnotation["Alpha"] = 0

        a = curAnnotation["Alpha"]

        dx = curAnnotation["h_par_cor"]*np.sin(a)/curAnnotation["pixelSpacingX"]
        dy = curAnnotation["h_par_cor"]*np.cos(a)/curAnnotation["pixelSpacingY"]
        curAnnotation["cor_LL_x"] = int(curAnnotation["cor_mid_x"] - dx)
        curAnnotation["cor_LL_y"] = int(curAnnotation["cor_mid_y"] + dy)
        curAnnotation["cor_LR_x"] = int(curAnnotation["cor_mid_x"] + dx)
        curAnnotation["cor_LR_y"] = int(curAnnotation["cor_mid_y"] - dy)

        # now the orthogonal direction
        dxu = curAnnotation["h_orth_cor"]*np.sin(a)/curAnnotation["pixelSpacingX"]
        dyu = curAnnotation["h_orth_cor"]*np.cos(a)/curAnnotation["pixelSpacingY"]
        curAnnotation["cor_U_mid_x"] = int(curAnnotation["cor_mid_x"] + dyu)
        curAnnotation["cor_U_mid_y"] = int(curAnnotation["cor_mid_y"] + dxu)

        curAnnotation["cor_UL_x"] = int(curAnnotation["cor_U_mid_x"] - dx)
        curAnnotation["cor_UL_y"] = int(curAnnotation["cor_U_mid_y"] + dy)
        curAnnotation["cor_UR_x"] = int(curAnnotation["cor_U_mid_x"] + dx)
        curAnnotation["cor_UR_y"] = int(curAnnotation["cor_U_mid_y"] - dy)


        #### TRANSVERSALE

        # now we need to walk 180/2 mm in either direction
        a = curAnnotation["Alpha"]

        dx = curAnnotation["h_par_tra"]*np.sin(a)/curAnnotation["pixelSpacingX"]
        dy = curAnnotation["h_par_tra"]*np.cos(a)/curAnnotation["pixelSpacingY"]
        curAnnotation["tra_LL_x"] = int(curAnnotation["tra_mid_x"] - dx)
        curAnnotation["tra_LL_y"] = int(curAnnotation["tra_mid_y"] + dy)
        curAnnotation["tra_LR_x"] = int(curAnnotation["tra_mid_x"] + dx)
        curAnnotation["tra_LR_y"] = int(curAnnotation["tra_mid_y"] - dy)

        # now the orthogonal direction
        dxu = curAnnotation["h_orth_tra"]*np.sin(a)/curAnnotation["pixelSpacingX"]
        dyu = curAnnotation["h_orth_tra"]*np.cos(a)/curAnnotation["pixelSpacingY"]
        curAnnotation["tra_U_mid_x"] = int(curAnnotation["tra_mid_x"] + 3*dyu/4)
        curAnnotation["tra_U_mid_y"] = int(curAnnotation["tra_mid_y"] + 3*dxu/4)

        curAnnotation["tra_UL_x"] = int(curAnnotation["tra_U_mid_x"] - dx)
        curAnnotation["tra_UL_y"] = int(curAnnotation["tra_U_mid_y"] + dy)
        curAnnotation["tra_UR_x"] = int(curAnnotation["tra_U_mid_x"] + dx)
        curAnnotation["tra_UR_y"] = int(curAnnotation["tra_U_mid_y"] - dy)

        curAnnotation["tra_L_mid_x"] = int(curAnnotation["tra_mid_x"] - 1*dyu/4)
        curAnnotation["tra_L_mid_y"] = int(curAnnotation["tra_mid_y"] - 1*dxu/4)

        curAnnotation["tra_LL_x"] = int(curAnnotation["tra_L_mid_x"] - dx)
        curAnnotation["tra_LL_y"] = int(curAnnotation["tra_L_mid_y"] + dy)
        curAnnotation["tra_LR_x"] = int(curAnnotation["tra_L_mid_x"] + dx)
        curAnnotation["tra_LR_y"] = int(curAnnotation["tra_L_mid_y"] - dy)

        curAnnotation["Dirty"] = False



def getPath (row):
    csvPath = '_'.join(row["path"].iloc[0].split("/")[-3:-1]) + ".csv"
    csvPath = os.path.join(annotationPath, csvPath)
    return csvPath


def loadAnnotation (row):
    csvPath = getPath(row)
    print ("Loading from ", csvPath)
    try:
        a = pd.read_csv(csvPath)
    except:
        print ("Could not read annotation. Ignoring")
        a = emptyAnnotation(row)
    return(a)


def saveAnnotation (row):
    csvPath = getPath(row)
    print ("Saving to", csvPath)
    row.to_csv(csvPath, index = False, header=True)


def emptyAnnotation (row):
    row = row.copy()
    row["curFrame"] = -1
    row["Dirty"] = False

    row["h_par_cor"] = 160/2
    row["h_orth_cor"] = 60
    row["h_par_tra"] = 60/2
    row["h_orth_tra"] = 160

    row["pixelSpacingX"] = -1
    row["pixelSpacingY"] = -1
    row["pixelSpacingZ"] = -1

    row["cor_LL_x"] = -1
    row["cor_LL_y"] = -1
    row["cor_LR_x"] = -1
    row["cor_LR_y"] = -1
    row["cor_UL_x"] = -1
    row["cor_UL_y"] = -1
    row["cor_UR_x"] = -1
    row["cor_UR_y"] = -1
    row["tra_LL_x"] = -1
    row["tra_LL_y"] = -1
    row["tra_LR_x"] = -1
    row["tra_LR_y"] = -1
    row["tra_UL_x"] = -1
    row["tra_UL_y"] = -1
    row["tra_UR_x"] = -1
    row["tra_UR_y"] = -1
    row["Deleted"] = 0
    return row



def loadMRSeries (mrFolder, seriesID = None, verbose = False):
    try:
        # first get all series
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(os.path.join(mrFolder))
        seriesID = series_ids[0]
        if verbose == True:
            print ("Found seriesID", seriesID)
        dicom_names = reader.GetGDCMSeriesFileNames(os.path.join(mrFolder), seriesID)
        if verbose == True:
            print (dicom_names)
        reader.SetFileNames(dicom_names)
        mrImgITK = reader.Execute()
    except Exception as e:
        print ("Failed. This should not happen. Error", e)
        print ("Trying to load meta nonetheless")
        meta = pydicom.dcmread(dicom_names[0])
        print (meta)
        raise Exception(e)
    return mrImgITK



def loadMR(row):
    # determine the BP image
    mrFolder = os.path.join(row["path"].iloc[0])
    mhas = glob (f"{mrFolder}/*sag*.mha")
    print (mhas)
    # we only want one mha here , else we throw, and continue with next
    if len(mhas) > 0:
        mrImgITK = sitk.ReadImage(mhas[0])
    else:
        # else there is a subfolder (a single one, please) with the dicoms
        mrFolder = glob (f"{mrFolder}/*/")[0]
        # read all series IDs
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(mrFolder)

        # search for BP
        loadSeries = None
        print ("Checking series", series_ids)
        assert (len(series_ids) == 1)

        dicom_names = reader.GetGDCMSeriesFileNames(mrFolder, series_ids[0])
        meta = pydicom.dcmread(dicom_names[0])
        seriesName = meta[0x0008,0x103e].value
        loadSeries = seriesName

        # load what we have found
        print ("Found series name", loadSeries)
        try:
            mrImgITK = loadMRSeries (mrFolder, loadSeries)
        except:
            print ("COULD NOT LOAD IT. Possible MPR Sag. Continue")
            return None, (None, None, None)

    spacing = mrImgITK.GetSpacing()
    mrImg = sitk.GetArrayFromImage(mrImgITK)
    neu = (mrImg-np.min(mrImg))/(np.max(mrImg)-np.min(mrImg)) * 255.0
    mrImg = np.asarray(neu, dtype = np.uint8)
    return mrImg, spacing



def loadMRandAnnotation(row, c = None):
    mrImg, spacing = loadMR (row)
    print (mrImg.shape)

    # try annotation
    annotation = loadAnnotation (row)
    return mrImg, annotation, spacing


def move_point_towards(point1, point2, distance):
    # Calculate the vector between point1 and point2
    vector = [point2[0] - point1[0], point2[1] - point1[1]]

    # Normalize the vector
    length = (vector[0]**2 + vector[1]**2)**0.5
    normalized_vector = [vector[0] / length, vector[1] / length]

    # Move point1 towards point2 by the specified distance
    new_point = [point1[0] + normalized_vector[0] * distance, point1[1] + normalized_vector[1] * distance]

    return new_point



def shrink_rectangle(current_annotation, shrink_distance, prefix):
    LL = [current_annotation[prefix+"_LL_x"], current_annotation[prefix+"_LL_y"]]
    LR = [current_annotation[prefix+"_LR_x"], current_annotation[prefix+"_LR_y"]]
    UL = [current_annotation[prefix+"_UL_x"], current_annotation[prefix+"_UL_y"]]
    UR = [current_annotation[prefix+"_UR_x"], current_annotation[prefix+"_UR_y"]]

    if prefix == "cor":
        new_UL = move_point_towards(UL, LL, shrink_distance)
        new_UR = move_point_towards(UR, LR, shrink_distance)
        current_annotation[prefix+"_UL_x"], current_annotation[prefix+"_UL_y"] = new_UL
        current_annotation[prefix+"_UR_x"], current_annotation[prefix+"_UR_y"] = new_UR
    else:
        new_UL = move_point_towards(UL, UR, shrink_distance)
        new_UR = move_point_towards(LL, LR, shrink_distance)
        current_annotation[prefix+"_UL_x"], current_annotation[prefix+"_UL_y"] = new_UL
        current_annotation[prefix+"_LL_x"], current_annotation[prefix+"_LL_y"] = new_UR

    return current_annotation



def move_rectangle(current_annotation, x_distance, y_distance, prefix):
    current_annotation[prefix+"_LL_x"] += x_distance
    current_annotation[prefix+"_LL_y"] += y_distance

    current_annotation[prefix+"_LR_x"] += x_distance
    current_annotation[prefix+"_LR_y"] += y_distance

    current_annotation[prefix+"_UL_x"] += x_distance
    current_annotation[prefix+"_UL_y"] += y_distance

    current_annotation[prefix+"_UR_x"] += x_distance
    current_annotation[prefix+"_UR_y"] += y_distance

    return current_annotation


def rotate_point_around_origin(point, angle_rad):
    # Rotate a point around the origin by a given angle (in radians)
    x = point[0]
    y = point[1]
    new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)
    return [new_x, new_y]


def rotate_rectangle(current_annotation, angle_degrees, prefix):
    # Convert angle from degrees to radians
    angle_rad = math.radians(angle_degrees)

    # Calculate the center of the rectangle
    center_x = (current_annotation[prefix+"_UL_x"] + current_annotation[prefix+"_LR_x"]) / 2
    center_y = (current_annotation[prefix+"_UL_y"] + current_annotation[prefix+"_LR_y"]) / 2

    # Translate all points so that the center is at the origin
    translated_UL = [current_annotation[prefix+"_UL_x"] - center_x, current_annotation[prefix+"_UL_y"] - center_y]
    translated_LR = [current_annotation[prefix+"_LR_x"] - center_x, current_annotation[prefix+"_LR_y"] - center_y]
    translated_UR = [current_annotation[prefix+"_UR_x"] - center_x, current_annotation[prefix+"_UR_y"] - center_y]
    translated_LL = [current_annotation[prefix+"_LL_x"] - center_x, current_annotation[prefix+"_LL_y"] - center_y]

    # Rotate all translated points
    rotated_UL = rotate_point_around_origin(translated_UL, angle_rad)
    rotated_LR = rotate_point_around_origin(translated_LR, angle_rad)
    rotated_UR = rotate_point_around_origin(translated_UR, angle_rad)
    rotated_LL = rotate_point_around_origin(translated_LL, angle_rad)

    # Translate back to the original position by adding the center coordinates
    current_annotation[prefix+"_UL_x"] = rotated_UL[0] + center_x
    current_annotation[prefix+"_UL_y"] = rotated_UL[1] + center_y

    current_annotation[prefix+"_LR_x"] = rotated_LR[0] + center_x
    current_annotation[prefix+"_LR_y"] = rotated_LR[1] + center_y

    current_annotation[prefix+"_UR_x"] = rotated_UR[0] + center_x
    current_annotation[prefix+"_UR_y"] = rotated_UR[1] + center_y

    current_annotation[prefix+"_LL_x"] = rotated_LL[0] + center_x
    current_annotation[prefix+"_LL_y"] = rotated_LL[1] + center_y

    return current_annotation


if  __name__ == '__main__':
    opt = BaseOptions().parse()
    cohort = opt.cohort

    if cohort == "train":
        MRPath = "./data/train/PICAI"
        annotationPath = "annotations/train"
        print ("Searching for training MR data (PICAI) in", os.path.join(MRPath, "*/"))

    elif "test." in cohort:
        cohort_name = cohort.replace('test.', '')
        MRPath = f"./data/test/{cohort_name}"
        annotationPath = f"annotations/test/{cohort_name}/"
        print (f"Searching for test MR data ({cohort_name}) in", os.path.join(MRPath, "*/"))
    else:
        raise Exception ("Unknown cohort.")

    mrList = glob(os.path.join(MRPath, "*/"))
    mrList = sorted(mrList)
    print ("Found", len(mrList), "candidates")

    os.makedirs (annotationPath, exist_ok = True)

    curMR = 0
    maxMR = len(mrList)

    while True:
        # ensure that index of current MR exists
        if curMR < 0:
            curMR = 0
        if curMR > maxMR - 1:
            curMR = maxMR - 1

        clipboard = None

        # erzeuge zeile einer tabelle
        row = pd.DataFrame.from_dict({"path": str(mrList[curMR])}, orient = "index").T

        try:
            print ("Loading MR and annotation...")
            print(row)
            imgStack, imgAnnotation, (pixelSpacingX, pixelSpacingY, pixelSpacingZ) = loadMRandAnnotation (row, cohort)
        except Exception as e:
            print(e)
            raise Exception ("MR does not exist. Trying next file")

        if imgStack is None:
            curMR = curMR + 1
            continue

        # OK now..
        if np.round(pixelSpacingX,2) != np.round(pixelSpacingY,2):
            raise Exception ("Non-uniform pixelspacing!")

        # coordinates
        maxFrame = imgStack.shape[0]
        curFrame = 0

        # jump to annotation

        # have any annotation, then take last one
        aFrames = sorted(imgAnnotation["curFrame"])
        if aFrames[-1] != -1:
            curFrame = aFrames[-1]

        print ("Starting at frame", curFrame)
        if curFrame < 0 or curFrame > maxFrame - 1:
            print ("ERROR, frame outside expectation!")
            curFrame = 0


        current = 0
        cv2.namedWindow ("MR", cv2.WINDOW_GUI_EXPANDED)
        lastOne = 0
        gotoNext = 0
        k = 0
        while 1 == 1:
            # get annotation of current frame
            curAnnotation = imgAnnotation.query("curFrame == @curFrame").copy()
            if len(curAnnotation) == 0:
                curAnnotation = emptyAnnotation(row)

            # we are missing Dirty in generated annotations
            if "Dirty" not in curAnnotation.keys():
                curAnnotation["Dirty"] = True

            # update pixelspacing in any case
            row["pixelSpacingX"] = pixelSpacingX
            row["pixelSpacingY"] = pixelSpacingY
            row["pixelSpacingZ"] = pixelSpacingZ

            image = np.stack((imgStack[curFrame,:,:],)*3, axis = -1)
            #print ("Imsize:", image.shape)
            cv2.setMouseCallback("MR", mouseCallback)

            overlayIm = None
            while k != ESC and k != SKIP and k != BACK and k != K_9:

                mousePressed = 255
                k = cv2.waitKey(30) & 0xFF
                #print(k)
                if mousePressed == 255 and k == 255 and overlayIm is not None and curAnnotation.iloc[0]["Dirty"] == False:
                    continue

                # restore mask
                overlayIm = image.copy()

                lineColor = (0,0,128)
                lineThickness = 1
                displayImg = overlayIm.copy()

                # enhance contrast
                factor = clipFactor
                clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(int(2*factor),int(2*factor)))
                tmpIm = clahe.apply(displayImg[:,:,2])

                displayImg[:,:,0] = tmpIm.copy()
                displayImg[:,:,1] = tmpIm.copy()
                displayImg[:,:,2] = tmpIm.copy()

                if curFrame == curAnnotation.iloc[0]["curFrame"]:
                    lineColor = bboxcolors["cor"]
                    cv2.line(displayImg, (int(curAnnotation["cor_LL_x"]), int(curAnnotation["cor_LL_y"])), (int(curAnnotation["cor_LR_x"]), int(curAnnotation["cor_LR_y"])), color = lineColor, thickness = lineThickness)
                    cv2.line(displayImg, (int(curAnnotation["cor_LL_x"]), int(curAnnotation["cor_LL_y"])), (int(curAnnotation["cor_UL_x"]), int(curAnnotation["cor_UL_y"])), color = lineColor, thickness = lineThickness)
                    cv2.line(displayImg, (int(curAnnotation["cor_LR_x"]), int(curAnnotation["cor_LR_y"])), (int(curAnnotation["cor_UR_x"]), int(curAnnotation["cor_UR_y"])), color = lineColor, thickness = lineThickness)
                    cv2.line(displayImg, (int(curAnnotation["cor_UR_x"]), int(curAnnotation["cor_UR_y"])), (int(curAnnotation["cor_UL_x"]), int(curAnnotation["cor_UL_y"])), color = lineColor, thickness = lineThickness)

                    lineColor = bboxcolors["tra"]
                    cv2.line(displayImg, (int(curAnnotation["tra_LL_x"]), int(curAnnotation["tra_LL_y"])), (int(curAnnotation["tra_LR_x"]), int(curAnnotation["tra_LR_y"])), color = lineColor, thickness = lineThickness)
                    cv2.line(displayImg, (int(curAnnotation["tra_LL_x"]), int(curAnnotation["tra_LL_y"])), (int(curAnnotation["tra_UL_x"]), int(curAnnotation["tra_UL_y"])), color = lineColor, thickness = lineThickness)
                    cv2.line(displayImg, (int(curAnnotation["tra_LR_x"]), int(curAnnotation["tra_LR_y"])), (int(curAnnotation["tra_UR_x"]), int(curAnnotation["tra_UR_y"])), color = lineColor, thickness = lineThickness)
                    cv2.line(displayImg, (int(curAnnotation["tra_UR_x"]), int(curAnnotation["tra_UR_y"])), (int(curAnnotation["tra_UL_x"]), int(curAnnotation["tra_UL_y"])), color = lineColor, thickness = lineThickness)

                try:
                    mrName = os.path.split(os.path.dirname(str(mrList[curMR])))[-1]
                except:
                    mrName = ''

                cv2.putText(displayImg, "S:" + str(curFrame) + "/" + str(maxFrame) + " -- " + "M:" + str(curMR) + "/" + str(maxMR) + " -- " + mrName, (0,16), font, 0.4, (0,0,255), 1)

                if curAnnotation.iloc[0]["curFrame"] != -1:
                    displayImg [:, 0, 1] = 255

                try:
                    if curAnnotation["Deleted"].iloc[0] == 1:
                        cv2.putText(displayImg, "DELETED", (0,displayImg.shape[0]//2), font, 3, (0, 0, 255), 1)
                except Exception as e:
                    print (curAnnotation)
                    print ("DELE", e)
                    pass


                cv2.imshow("MR", displayImg)


                def updateImgAnn (imgAnnotation, curAnnotation):
                    frameNumber = curAnnotation.iloc[0]["curFrame"]
                    if frameNumber in imgAnnotation['curFrame'].values:
                        imgAnnotation = imgAnnotation.query("curFrame != @frameNumber").copy()
                    imgAnnotation = pd.concat([imgAnnotation, curAnnotation], ignore_index=True).reset_index(drop = True)
                    return imgAnnotation


                def deleteImgAnn (imgAnnotation, frameNumber):
                    if frameNumber != -1:
                        imgAnnotation = imgAnnotation.query("curFrame != @frameNumber").copy()
                    return imgAnnotation



                if k == K_x:
                    curFrame = curFrame + 1
                    if curAnnotation.iloc[0]["curFrame"] != -1:
                        imgAnnotation = updateImgAnn(imgAnnotation, curAnnotation)
                    break
                if k == K_z:
                    curFrame = curFrame - 1
                    if curAnnotation.iloc[0]["curFrame"] != -1:
                        imgAnnotation = updateImgAnn(imgAnnotation, curAnnotation)
                    break
                if k == K_o:
                    clipLimit = clipLimit * 2
                    if clipLimit > 32:
                        clipLimit = 1
                if k == K_p:
                    clipFactor = clipFactor * 2
                    if clipFactor > 32:
                        clipFactor = 1

                if k == K_q:
                    # verkleinern in orthogonaler richtung
                    curAnnotation = shrink_rectangle(curAnnotation, 2, "cor")
                    curAnnotation["Dirty"] = True

                if k == K_e:
                    # vergroessern in orthogonaler richtung
                    curAnnotation = shrink_rectangle(curAnnotation, -2, "cor")
                    curAnnotation["Dirty"] = True

                if k == K_a:
                    # verkleinern in orthogonaler richtung
                    curAnnotation = shrink_rectangle(curAnnotation, 2, "tra")
                    curAnnotation["Dirty"] = True

                if k == K_d:
                    # vergroessern in orthogonaler richtung
                    curAnnotation = shrink_rectangle(curAnnotation, -2, "tra")
                    curAnnotation["Dirty"] = True


                if k == K_h:
                    curAnnotation = move_rectangle(curAnnotation, 2, 0, "tra")
                    curAnnotation["Dirty"] = True

                if k == K_f:
                    curAnnotation = move_rectangle(curAnnotation, -2, 0, "tra")
                    curAnnotation["Dirty"] = True

                if k == K_t:
                    curAnnotation = move_rectangle(curAnnotation, 0, -2, "tra")
                    curAnnotation["Dirty"] = True

                if k == K_g:
                    curAnnotation = move_rectangle(curAnnotation, 0, 2, "tra")
                    curAnnotation["Dirty"] = True


                if k == K_l:
                    curAnnotation = move_rectangle(curAnnotation, 2, 0, "cor")
                    curAnnotation["Dirty"] = True

                if k == K_j:
                    curAnnotation = move_rectangle(curAnnotation, -2, 0, "cor")
                    curAnnotation["Dirty"] = True

                if k == K_i:
                    curAnnotation = move_rectangle(curAnnotation, 0, -2, "cor")
                    curAnnotation["Dirty"] = True

                if k == K_k:
                    curAnnotation = move_rectangle(curAnnotation, 0, 2, "cor")
                    curAnnotation["Dirty"] = True


                if k == K_m:
                    curAnnotation = rotate_rectangle(curAnnotation, 1, "cor")
                    curAnnotation = rotate_rectangle(curAnnotation, 1, "tra")
                    curAnnotation["Dirty"] = True

                if k == K_n:
                    curAnnotation = rotate_rectangle(curAnnotation, -1, "cor")
                    curAnnotation = rotate_rectangle(curAnnotation, -1, "tra")
                    curAnnotation["Dirty"] = True

                if k == K_comma:
                    clipboard = curAnnotation.copy()

                if k == K_semicolon:
                    if clipboard is not None:
                        curAnnotation = clipboard.copy()
                        curAnnotation["curFrame"] = curFrame
                        curAnnotation["Dirty"] = True

                if k == DELETE:
                    imgAnnotation = deleteImgAnn(imgAnnotation, curAnnotation.iloc[0]["curFrame"])
                    curAnnotation = emptyAnnotation(row)
                    curAnnotation["Dirty"] = True

                if k == BACK:
                    if curAnnotation.iloc[0]["curFrame"] != -1:
                        imgAnnotation = updateImgAnn(imgAnnotation, curAnnotation)
                    saveAnnotation (imgAnnotation)
                    print ("going back.")
                    gotoNext = -1
                    break

                if k == SKIP:
                    if curAnnotation.iloc[0]["curFrame"] != -1:
                        imgAnnotation = updateImgAnn(imgAnnotation, curAnnotation)
                    saveAnnotation (imgAnnotation)
                    gotoNext = 1
                    break

                if k == K_7:
                    if curAnnotation.iloc[0]["curFrame"] != -1:
                        imgAnnotation = updateImgAnn(imgAnnotation, curAnnotation)
                    saveAnnotation (imgAnnotation)
                    gotoNext = -25
                    break

                if k == K_9:
                    if curAnnotation.iloc[0]["curFrame"] != -1:
                        imgAnnotation = updateImgAnn(imgAnnotation, curAnnotation)
                    saveAnnotation (imgAnnotation)
                    gotoNext = 25
                    break

                if k == 27:
                    if curAnnotation.iloc[0]["curFrame"] != -1:
                        imgAnnotation = updateImgAnn(imgAnnotation, curAnnotation)
                    saveAnnotation (imgAnnotation)
                    print ("Stopping.")
                    exit(-1)
            if curFrame < 0:
                curFrame = 0
            if curFrame > maxFrame-1:
                curFrame =  maxFrame -1
            if gotoNext != 0:
                break
        curMR = curMR + gotoNext

    cv2.destroyAllWindows()

#
