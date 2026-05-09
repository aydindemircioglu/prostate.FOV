import mmrotate
from joblib import dump, load
import cv2
import pandas as pd
import numpy as np
import os
from glob import glob
import joblib
import subprocess
from pathlib import Path
import pickle
import sys

sys.path.append("..")
from utils_path import recreatePath
from prepareTrainData import getAllAnnotations


frame_size = 6
resize_size = 160
border_size = 32


def visualizePreds (fpkl, oPath):
    recreatePath (oPath)
    data = joblib.load(fpkl)
    for k in range(len(data)):
        preds = data[k]
        fpng = preds["meta"][0].data["filename"]
        # have two classes
        try:
            slice = mmrotate.core.bbox.obb2poly_np(preds["pred"][0], version='le90')[0] # box with highest prob
        except:
            sliceErrors.append(20)
            continue

        fann = fpng.replace("/images/", "/annotations/")
        fann = fann.replace(".png", ".txt")
        gtann =  pd.read_csv(fann, header = None)

        print (gtann)
        exit(-1)
        # paint gt ann first
        img = cv2.imread(fpng)

        # greyscale, ignore pseudorgb
        img[:,:,0] = img[:,:,1]
        img[:,:,2] = img[:,:,1]

        # add some extra margin to see better
        canvas = np.zeros((img.shape[0]+512,img.shape[1]+512,3), dtype = np.uint8)
        canvas[256:256+img.shape[0], 256:256+img.shape[1], :] = img

        pcolor = {"box": (255, 0, 0), "gtbox": (0, 255,0)}
        for j in range(len(gtann)):
            # move the gt box two pixels to be better visible
            z  = [int(k)+2 for k in gtann.iloc[j][0].split(" ")[0:8]]
            cv2.line (canvas, (int(z[0])+256, int(z[1])+256), (int(z[2])+256, int(z[3])+256), pcolor["gtbox"], 2)
            cv2.line (canvas, (int(z[2])+256, int(z[3])+256), (int(z[4])+256, int(z[5])+256), pcolor["gtbox"], 2)
            cv2.line (canvas, (int(z[4])+256, int(z[5])+256), (int(z[6])+256, int(z[7])+256), pcolor["gtbox"], 2)
            cv2.line (canvas, (int(z[6])+256, int(z[7])+256), (int(z[0])+256, int(z[1])+256), pcolor["gtbox"], 2)

            # paint cor and tra
            z = slice
            cv2.line (canvas, (int(z[0])+256, int(z[1])+256), (int(z[2])+256, int(z[3])+256), pcolor["box"], 2)
            cv2.line (canvas, (int(z[2])+256, int(z[3])+256), (int(z[4])+256, int(z[5])+256), pcolor["box"], 2)
            cv2.line (canvas, (int(z[4])+256, int(z[5])+256), (int(z[6])+256, int(z[7])+256), pcolor["box"], 2)
            cv2.line (canvas, (int(z[6])+256, int(z[7])+256), (int(z[0])+256, int(z[1])+256), pcolor["box"], 2)

        canvas = canvas[256:256+img.shape[0], 256:256+img.shape[1],:]

        # selected slicesas well
        x0, y0, x1, y1, x2, y2, x3, y3, _= slice
        pred_bx = int(x0+x1+x2+x3)//4
        pred_by = int(y0+y1+y2+y3)//4

        pred_slice = pred_bx//128 + pred_by//128*8

        pred_bx = pred_bx//128
        pred_by = pred_by//128

        red = (0,0,255) #BGR
        cv2.line(canvas, (pred_bx*128, pred_by*128), (pred_bx*128+128, pred_by*128), red, 3)
        cv2.line(canvas, (pred_bx*128, pred_by*128), (pred_bx*128, pred_by*128+128), red, 3)
        cv2.line(canvas, (pred_bx*128+128, pred_by*128), (pred_bx*128+128, pred_by*128+128), red, 3)
        cv2.line(canvas, (pred_bx*128, pred_by*128+128), (pred_bx*128+128, pred_by*128+128), red, 3)

        cv2.imwrite (os.path.join(oPath, os.path.basename(fpng)), canvas)

#



if __name__ == '__main__':
    pklDir = "../data/stage1/checkpoints/"

    # we need to reload the annotations to decide which box was the true one
    allAnn = getAllAnnotations()
    allAnn = pd.concat(allAnn).reset_index(drop = True)
    allAnn = allAnn.sort_values(["path"]).copy()
    # we have _mid_ keys, these are not ok, not clear how they made it into the annotation
    allAnn = allAnn.drop([k for k in allAnn.keys() if "_mid" in k or "_ort" in k or "_par" in k or "Alpha" in k], axis = 1).copy()

    # we have no deleted, but nonetheless
    allAnn = allAnn.query("Deleted == 0").copy()
    allAnn = allAnn.query("cor_LR_y > 0").copy()
    allAnn = allAnn.query("cor_UR_y > 0").copy()

    # we split by patient, so we need this
    allAnn["AccNr"] = [k.split("/")[-2] for k in allAnn["path"]]

    medslices = []
    results = {}
    missings = {}
    for fpkl in glob (os.path.join(pklDir, "*/*.pkl")):
        data = load(fpkl)
        print (fpkl)

        sliceErrors = []
        curmissings = 0
        for k in range(len(data)):
            preds = data[k]
            png = preds["meta"][0].data["filename"]
            # have two classes
            try:
                slice = mmrotate.core.bbox.obb2poly_np(preds["pred"][0], version='le90')[0] # box with highest prob
            except:
                slice = None
                curmissings = curmissings + 1

            fann = png.replace("/images/", "/annotations/")
            fann = fann.replace(".png", ".txt")
            accNr = fann.split("_")[-2]
            gtann =  pd.read_csv(fann, header = None)

            patFrames = allAnn.query("AccNr == @accNr").copy()
            patFrames = patFrames.sort_values(["curFrame"])
            sliceFrame = patFrames.iloc[len(patFrames)//2]
            gt_slice = sliceFrame["curFrame"]
            medslices.append(gt_slice)

            # determine predicted slice
            if slice is None:
                pred_slice = 10 # corresponds to median of the gt slices
            else:
                x0, y0, x1, y1, x2, y2, x3, y3, _= slice
                pred_bx = int(x0+x1+x2+x3)//4
                pred_by = int(y0+y1+y2+y3)//4
                pred_bx = pred_bx//(resize_size+border_size)
                pred_by = pred_by//(resize_size+border_size)
                pred_slice = pred_bx + pred_by*frame_size

            # determine minimal diff
            diffs = np.abs(np.array(gt_slice) - pred_slice)
            d = np.min(diffs)
            sliceErrors.append(d)

        #print (fpkl, np.mean(dices), "+/-", np.std(dices))
        results[fpkl] = sliceErrors
        missings[fpkl] = curmissings

    # merge folds
    final = {}
    finalmissings = {}
    for k in results.keys():
        model = os.path.basename(k).replace(".pkl", "").split("_")
        LR = model[-1]
        fold = model[-2] # ignore
        arch = '_'.join(model[0:-2])
        modelname = arch+"_"+LR
        if  arch+"_"+LR in final:
            final[modelname].extend(results[k])
            finalmissings[modelname] += missings[k]
        else:
            final[modelname] = results[k]
            finalmissings[modelname] = missings[k]

    df = []
    for k in final.keys():
        #assert  (len(final[k]) == 1179)
        z = np.array(final[k])
        #missings = np.sum(z == 99) # should be zero, because now we predict slice 10
        MAE = np.round(np.mean(z[z<99]),2)
        deviations0 = np.sum( z == 0 )
        deviations1 = np.sum( (z > 0) & (z < 99) )
        deviations2 = np.sum( (z > 1) & (z < 99) )
        deviations3p = np.sum( (z > 2) & (z < 99) )
        dev0p = np.round(deviations0/len(z)*100,3)
        dev1p = np.round(deviations1/len(z)*100,3)
        dev2p = np.round(deviations2/len(z)*100,3)
        dev3pp = np.round(deviations3p/len(z)*100,3)
        df.append( {"Model": k, "Error": np.round(np.mean(z), 2), "Std": np.round(np.std(z), 2)} )
        print (k,np.round(np.mean(z[z<99]), 4), "+/-", np.round(np.std(z[z<99]), 4), "E:", MAE, "NA:", finalmissings[k])
        print("\t", "DEV0:", deviations0, f"({dev0p}%)",
                    "DEV1:", deviations1, f"({dev1p}%)",
                    "DEV2:", deviations2, f"({dev2p}%)",
                    "DEV2+:",deviations3p, f"({dev3pp}%)")
    df = pd.DataFrame(df)
    df = df.sort_values(["Error"], ascending = True)
    print(df)
    os.makedirs("../results/", exist_ok = True)
    df.to_csv("../results/cv_results_stage1.csv", index = False)

    bestModel = df.iloc[0]["Model"]
    LR = bestModel.split("_")[-1]
    mName = '_'.join(bestModel.split("_")[0:-1])
    cfg = mName+".py"

    os.makedirs("../standalone/models/", exist_ok = True)
    execStr = f"cp ./configs/{mName}.py ../standalone/models/stage1_cfg.py"
    subprocess.call(execStr, shell=True)

    print ("Copying over model", bestModel)
    for f in range(5):
        execStr = f"cp ../data/stage1/checkpoints/{mName}/{mName}_{f}_{LR}/latest.pth ../standalone/models/stage1_f{f}.pth"
        subprocess.call(execStr, shell=True)

        # generate visualizations as well
        # pklfile = f"../data/stage1/checkpoints/{mName}/{mName}_{f}_{LR}.pkl"
        # visualizePreds (pklfile, f"./predictions/{mName}_{f}_{LR}")

#
