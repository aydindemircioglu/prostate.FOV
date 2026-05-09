import mmrotate
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import os
from glob import glob
import subprocess
import joblib
from joblib import parallel_backend, Parallel, delayed, load, dump

import sys
sys.path.append("..")
from utils_path import recreatePath



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




def drawAnnImg (oimg, curAnnotation, lineColor):
    xMaxC = np.max( [curAnnotation[k] for k in ["cor_LL_x", "cor_LR_x", "cor_UL_x", "cor_UR_x"]] )
    xMaxT = np.max( [curAnnotation[k] for k in ["tra_LL_x", "tra_LR_x", "tra_UL_x", "tra_UR_x"]] )
    xMax = np.max([xMaxC, xMaxT]) + 10 # safety

    yMaxC = np.max( [curAnnotation[k] for k in ["cor_LL_y", "cor_LR_y", "cor_UL_y", "cor_UR_y"]] )
    yMaxT = np.max( [curAnnotation[k] for k in ["tra_LL_y", "tra_LR_y", "tra_UL_y", "tra_UR_y"]] )
    yMax = np.max([yMaxC, yMaxT]) + 10 # safety

    lineThickness = 2
    ofs = 256
    _ = cv2.line(oimg, (int(curAnnotation["cor_LL_x"])-ofs, int(curAnnotation["cor_LL_y"])-ofs), (int(curAnnotation["cor_LR_x"])-ofs, int(curAnnotation["cor_LR_y"])-ofs), color = lineColor, thickness = lineThickness)
    _ = cv2.line(oimg, (int(curAnnotation["cor_LL_x"])-ofs, int(curAnnotation["cor_LL_y"])-ofs), (int(curAnnotation["cor_UL_x"])-ofs, int(curAnnotation["cor_UL_y"])-ofs), color = lineColor, thickness = lineThickness)
    _ = cv2.line(oimg, (int(curAnnotation["cor_LR_x"])-ofs, int(curAnnotation["cor_LR_y"])-ofs), (int(curAnnotation["cor_UR_x"])-ofs, int(curAnnotation["cor_UR_y"])-ofs), color = lineColor, thickness = lineThickness)
    _ = cv2.line(oimg, (int(curAnnotation["cor_UR_x"])-ofs, int(curAnnotation["cor_UR_y"])-ofs), (int(curAnnotation["cor_UL_x"])-ofs, int(curAnnotation["cor_UL_y"])-ofs), color = lineColor, thickness = lineThickness)

    lineColor = [l>>1 for l in lineColor]
    _ = cv2.line(oimg, (int(curAnnotation["tra_LL_x"])-ofs, int(curAnnotation["tra_LL_y"])-ofs), (int(curAnnotation["tra_LR_x"])-ofs, int(curAnnotation["tra_LR_y"])-ofs), color = lineColor, thickness = lineThickness)
    _ = cv2.line(oimg, (int(curAnnotation["tra_LL_x"])-ofs, int(curAnnotation["tra_LL_y"])-ofs), (int(curAnnotation["tra_UL_x"])-ofs, int(curAnnotation["tra_UL_y"])-ofs), color = lineColor, thickness = lineThickness)
    _ = cv2.line(oimg, (int(curAnnotation["tra_LR_x"])-ofs, int(curAnnotation["tra_LR_y"])-ofs), (int(curAnnotation["tra_UR_x"])-ofs, int(curAnnotation["tra_UR_y"])-ofs), color = lineColor, thickness = lineThickness)
    _ = cv2.line(oimg, (int(curAnnotation["tra_UR_x"])-ofs, int(curAnnotation["tra_UR_y"])-ofs), (int(curAnnotation["tra_UL_x"])-ofs, int(curAnnotation["tra_UL_y"])-ofs), color = lineColor, thickness = lineThickness)
    return (oimg)



def showAnnImg (png, curAnnotation, gtAnn):
    oimg = cv2.imread(png)
    #oimg = cv2.cvtColor(oimg, cv2.COLOR_GRAY2RGB)
    oimg = drawAnnImg (oimg, curAnnotation, (255, 0, 0))
    oimg = drawAnnImg (oimg, gtAnn, (255, 0, 255))
    oimg = cv2.resize(oimg, (1024, int(1024 * oimg.shape[0] / oimg.shape[1])), interpolation = cv2.INTER_AREA)
    cv2.imshow('img', oimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return (oimg)



# fixes size canvas of sie 2048x2048
def drawBoxes (curAnnotation, fill = True):
    xMaxC = np.max( [curAnnotation[k] for k in ["cor_LL_x", "cor_LR_x", "cor_UL_x", "cor_UR_x"]] )
    xMaxT = np.max( [curAnnotation[k] for k in ["tra_LL_x", "tra_LR_x", "tra_UL_x", "tra_UR_x"]] )
    xMax = np.max([xMaxC, xMaxT]) + 10 # safety

    yMaxC = np.max( [curAnnotation[k] for k in ["cor_LL_y", "cor_LR_y", "cor_UL_y", "cor_UR_y"]] )
    yMaxT = np.max( [curAnnotation[k] for k in ["tra_LL_y", "tra_LR_y", "tra_UL_y", "tra_UR_y"]] )
    yMax = np.max([yMaxC, yMaxT]) + 10 # safety

    yMax, xMax = 2048, 2048
    displayImgA = np.zeros((yMax, xMax, 3), dtype = np.uint8)
    lineColor = (255, 0, 0); lineThickness = 1
    _ = cv2.line(displayImgA, (int(curAnnotation["cor_LL_x"]), int(curAnnotation["cor_LL_y"])), (int(curAnnotation["cor_LR_x"]), int(curAnnotation["cor_LR_y"])), color = lineColor, thickness = lineThickness)
    _ = cv2.line(displayImgA, (int(curAnnotation["cor_LL_x"]), int(curAnnotation["cor_LL_y"])), (int(curAnnotation["cor_UL_x"]), int(curAnnotation["cor_UL_y"])), color = lineColor, thickness = lineThickness)
    _ = cv2.line(displayImgA, (int(curAnnotation["cor_LR_x"]), int(curAnnotation["cor_LR_y"])), (int(curAnnotation["cor_UR_x"]), int(curAnnotation["cor_UR_y"])), color = lineColor, thickness = lineThickness)
    _ = cv2.line(displayImgA, (int(curAnnotation["cor_UR_x"]), int(curAnnotation["cor_UR_y"])), (int(curAnnotation["cor_UL_x"]), int(curAnnotation["cor_UL_y"])), color = lineColor, thickness = lineThickness)
    if fill == True:
        seed_point = ( (int(curAnnotation["cor_LL_x"]) + int(curAnnotation["cor_UR_x"]))//2,
                       (int(curAnnotation["cor_LL_y"]) + int(curAnnotation["cor_UR_y"]))//2 )
        _ = cv2.floodFill(displayImgA, None, seedPoint=seed_point, newVal=lineColor)#, loDiff=(0, 0, 0, 0), upDiff=(0, 0, 0, 0))

    displayImgB = np.zeros((yMax, xMax, 3), dtype = np.uint8)
    lineColor = (0, 0, 255); lineThickness = 1
    cv2.line(displayImgB, (int(curAnnotation["tra_LL_x"]), int(curAnnotation["tra_LL_y"])), (int(curAnnotation["tra_LR_x"]), int(curAnnotation["tra_LR_y"])), color = lineColor, thickness = lineThickness)
    cv2.line(displayImgB, (int(curAnnotation["tra_LL_x"]), int(curAnnotation["tra_LL_y"])), (int(curAnnotation["tra_UL_x"]), int(curAnnotation["tra_UL_y"])), color = lineColor, thickness = lineThickness)
    cv2.line(displayImgB, (int(curAnnotation["tra_LR_x"]), int(curAnnotation["tra_LR_y"])), (int(curAnnotation["tra_UR_x"]), int(curAnnotation["tra_UR_y"])), color = lineColor, thickness = lineThickness)
    cv2.line(displayImgB, (int(curAnnotation["tra_UR_x"]), int(curAnnotation["tra_UR_y"])), (int(curAnnotation["tra_UL_x"]), int(curAnnotation["tra_UL_y"])), color = lineColor, thickness = lineThickness)
    if fill == True:
        seed_point = ( (int(curAnnotation["tra_LL_x"]) + int(curAnnotation["tra_UR_x"]))//2,
                       (int(curAnnotation["tra_LL_y"]) + int(curAnnotation["tra_UR_y"]))//2 )
        _ = cv2.floodFill(displayImgB, None, seedPoint=seed_point, newVal=lineColor)#, loDiff=(0, 0, 0, 0), upDiff=(0, 0, 0, 0))

    z = (displayImgA[:,:,0] == 255) & (displayImgB[:,:,2] == 255)
    return (z)



def visualizePreds (fpkl, oPath):
    recreatePath (oPath)
    data = joblib.load(fpkl)

    for k in range(len(data)):
        preds = data[k]
        png = preds["meta"][0].data["filename"]
        # have two classes
        try:
            cor = mmrotate.core.bbox.obb2poly_np(preds["pred"][0], version='le90')[0] # box with highest prob
            tra = mmrotate.core.bbox.obb2poly_np(preds["pred"][1], version='le90')[0] # box with highest prob
        except:
            # create annotation with no predictions
            oimg = cv2.imread(png)
            oimg[0:64, 1] = 255
            oimg[0:64, 0] = 255
            oimg[-64:, 0] = 255
            oimg[-64:, 1] = 255
            oimg = cv2.resize(oimg, (1024, int(1024 * oimg.shape[0] / oimg.shape[1])), interpolation = cv2.INTER_AREA)
            cv2.imwrite (os.path.join(oPath, os.path.basename(png)), oimg)
            continue

        fann = png.replace("/images/", "/annotations/")
        fann = fann.replace(".png", ".txt")
        gtann =  pd.read_csv(fann, header = None)

        # correct raw annotation
        cor, tra = correctAnnotations(cor, tra)
        corPred, traPred = cor, tra
        ann, gtann, corGT, traGT = extractAnnotations (cor, tra, gtann)

        # paint gt ann first
        oimg = cv2.imread(png)
        oimg = drawAnnImg (oimg, ann, (255, 0, 0))
        oimg = drawAnnImg (oimg, gtann, (255, 0, 255))
        oimg = cv2.resize(oimg, (1024, int(1024 * oimg.shape[0] / oimg.shape[1])), interpolation = cv2.INTER_AREA)
        cv2.imwrite (os.path.join(oPath, os.path.basename(png)), oimg)




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



def correctAnnotations(cor, tra):
    cvt_tra = [(tra[i], tra[i+1]) for i in range(0, len(tra)-1, 2)]
    cvt_cor = [(cor[i], cor[i+1]) for i in range(0, len(cor)-1, 2)]
    cvt_tra = fix_coordinates(cvt_cor, cvt_tra, False)
    tra = [item for tup in cvt_tra for item in tup] + [tra[-1]]
    return cor, tra


def computeAlpha(original_coordinates, perturbed_coordinates):
    original_coordinates = [(original_coordinates[i], original_coordinates[i+1]) for i in range(0, len(original_coordinates)-1, 2)]
    perturbed_coordinates = [(perturbed_coordinates[i], perturbed_coordinates[i+1]) for i in range(0, len(perturbed_coordinates)-1, 2)]

    # = GT
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

    # Calculate the average angle difference
    average_angle_diff = np.mean(angle_differences)
    return average_angle_diff



def extractAnnotations (cor, tra, gtann):
    # pred
    ann = {}
    ann["cor_LL_x"], ann["cor_LL_y"] = int(cor[0])+256, int(cor[1])+256
    ann["cor_LR_x"], ann["cor_LR_y"] = int(cor[2])+256, int(cor[3])+256
    ann["cor_UR_x"], ann["cor_UR_y"] = int(cor[4])+256, int(cor[5])+256
    ann["cor_UL_x"], ann["cor_UL_y"] = int(cor[6])+256, int(cor[7])+256

    ann["tra_LL_x"], ann["tra_LL_y"] = int(tra[0])+256, int(tra[1])+256
    ann["tra_LR_x"], ann["tra_LR_y"] = int(tra[2])+256, int(tra[3])+256
    ann["tra_UR_x"], ann["tra_UR_y"] = int(tra[4])+256, int(tra[5])+256
    ann["tra_UL_x"], ann["tra_UL_y"] = int(tra[6])+256, int(tra[7])+256


    tra = gtann.iloc[0][0].split(" ")
    cor = gtann.iloc[1][0].split(" ")
    if tra[-2] != "tra":
        tra, cor = cor, tra
    assert(tra[-2] == "tra")
    assert(cor[-2] == "cor")
    tra = [float(k) for k in tra[0:8]]
    cor = [float(k) for k in cor[0:8]]
    corGT, traGT = cor, tra
    gtann = {}
    gtann["cor_LL_x"], gtann["cor_LL_y"] = int(cor[0])+256, int(cor[1])+256
    gtann["cor_LR_x"], gtann["cor_LR_y"] = int(cor[2])+256, int(cor[3])+256
    gtann["cor_UR_x"], gtann["cor_UR_y"] = int(cor[4])+256, int(cor[5])+256
    gtann["cor_UL_x"], gtann["cor_UL_y"] = int(cor[6])+256, int(cor[7])+256

    gtann["tra_LL_x"], gtann["tra_LL_y"] = int(tra[0])+256, int(tra[1])+256
    gtann["tra_LR_x"], gtann["tra_LR_y"] = int(tra[2])+256, int(tra[3])+256
    gtann["tra_UR_x"], gtann["tra_UR_y"] = int(tra[4])+256, int(tra[5])+256
    gtann["tra_UL_x"], gtann["tra_UL_y"] = int(tra[6])+256, int(tra[7])+256
    return ann, gtann, corGT, traGT   # stupid...



def computeStats(fpkl):
    data = load(fpkl)
    print (fpkl)

    ious = []
    alphas = []
    for k in range(len(data)):
        preds = data[k]
        png = preds["meta"][0].data["filename"]
        # have two classes
        try:
            cor = mmrotate.core.bbox.obb2poly_np(preds["pred"][0], version='le90')[0] # box with highest prob
            tra = mmrotate.core.bbox.obb2poly_np(preds["pred"][1], version='le90')[0] # box with highest prob
        except:
            ious.append(-1)
            alphas.append(-1)
            continue

        fann = png.replace("/images/", "/annotations/")
        fann = fann.replace(".png", ".txt")
        gtann =  pd.read_csv(fann, header = None)

        # correct raw annotation
        cor, tra = correctAnnotations(cor, tra)
        corPred, traPred = cor, tra
        ann, gtann, corGT, traGT = extractAnnotations (cor, tra, gtann)

        try:
            imA = drawBoxes(ann)
            imB = drawBoxes(gtann)
            displayPreds = False
            if displayPreds == True:
                showAnnImg (png, ann, gtann)
        except Exception as e:
            print(e)
            print(imA.shape)
            print(canvA.shape)
            raise(e)
            ious.append(-1)
            alphas.append(-1)
            continue

        ious.append(iou(imA, imB))
        alphas.append(np.abs(computeAlpha(corGT, corPred)))

    print (np.mean(alphas))
    print (np.mean(ious))
    return fpkl, ious, alphas



if __name__ == '__main__':
    pklDir = "../data/stage2/checkpoints/"
    results = {}

    ncpus = 26
    ffilter = os.path.join(pklDir, "*/*.pkl")
    with parallel_backend("loky", inner_max_num_threads=1):
        res = Parallel (n_jobs = ncpus)(delayed(computeStats)(fpkl) for fpkl in glob (ffilter))
    print (res)
    for z in res:
        results[z[0]] = (z[1], z[2])

    # identify best

    # merge folds
    final = {}
    for k in results.keys():
        model = os.path.basename(k).replace(".pkl", "").split("_")
        LR = model[-1]
        fold = model[-2] # ignore
        arch = '_'.join(model[0:-2])
        modelname = arch+"_"+LR
        if  arch+"_"+LR in final:
            final[modelname] = tuple(a+b for a, b in zip(final[modelname], results[k]))
        else:
            final[modelname] = results[k]

    df = []
    for k in final.keys():
        missings = np.sum(np.array(final[k]) == -1)
        ciou = final[k][0]
        calpha = final[k][1]
        closs = 100*np.array(ciou) + (90-np.array(calpha))*2
        df.append( {"Model": k,
                "IoU": np.round(np.mean(ciou), 4),
                "IoU_Std": np.round(np.std(ciou), 2),
                "Alpha": np.round(np.mean(calpha), 4),
                "Alpha_Std": np.round(np.std(calpha), 2),
                "CombLoss": np.round(np.mean(closs), 4),
                "Miss": missings} )

    df = pd.DataFrame(df)
    print(df)
    df = df.sort_values(["CombLoss"], ascending = False)
    df.to_csv("../results/cv_results_iou.csv", index = False)

    bestModel = df.iloc[0]["Model"]
    LR = bestModel.split("_")[-1]
    mName = '_'.join(bestModel.split("_")[0:-1])
    cfg = mName+".py"

    execStr = f"cp ./configs/{mName}.py ../standalone/models/stage2_cfg.py"
    subprocess.call(execStr, shell=True)

    print ("Copying over model", bestModel)
    for f in range(5):
        execStr = f"cp /data/data/prostata.roi/stage2/checkpoints/{mName}/{mName}_{f}_{LR}/latest.pth ../standalone/models/stage2_f{f}.pth"
        subprocess.call(execStr, shell=True)

        # generate visualizations as well
        # pklfile = f"/data/data/prostata.roi/stage2/checkpoints/{mName}/{mName}_{f}_{LR}.pkl"
        # visualizePreds (pklfile, f"./predictions/{mName}_{f}_{LR}")


#
