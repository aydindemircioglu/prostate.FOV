#!/usr/bin/python3
import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
import cv2
from glob import glob
import pydicom

import sys
sys.path.append("..")
from utils_montage import *
from annotate import loadMR

import mmcv
from mmengine.runner import load_checkpoint

from mmdet.apis import inference_detector
from mmrotate.models import build_detector
import mmrotate


frame_size = 6
resize_size = 160
border_size = 32


def exportMontage (mrImg, fMontageImage):
    try:
        mrImg = sitk.GetArrayFromImage(mrImg)
        neu = (mrImg-np.min(mrImg))/(np.max(mrImg)-np.min(mrImg)) * 255.0
        mrImg = np.asarray(neu, dtype = np.uint8)
    except:
        pass # should be a numpy anyway

    montage, _ = getMontage(mrImg, frame_offset = 0)
    cv2.imwrite(fMontageImage, montage)
    return montage



def loadModel (device = "cuda:0", stage = None, fold = None):
    fcfg = "models/stage1_cfg.py"
    fstageckpt = f"./models/stage1_f{fold}.pth"

    config = mmcv.Config.fromfile(fcfg)
    config.model.pretrained = None

    def fixModel (cfg):
        print (cfg.model.type)
        if cfg.model.type == "CascadeRCNN":
            for k in range(len(cfg.model.roi_head.bbox_head)):
                cfg.model.roi_head.bbox_head[k]["num_classes"] = 1
        elif cfg.model.type == "RotatedRetinaNet":
            cfg.model.bbox_head["num_classes"] = 1
        elif cfg.model.type == "ReDet":
            for k in range(len(cfg.model.roi_head.bbox_head)):
                cfg.model.roi_head.bbox_head[k]["num_classes"] = 1
        elif cfg.model.type == "RoITransformer":
            for k in range(len(cfg.model.roi_head.bbox_head)):
                cfg.model.roi_head.bbox_head[k]["num_classes"] = 1
        else:
            raise Exception ("Unknown model, cannot adapt number of classes.")
        return cfg
    config = fixModel(config)

    model = build_detector(config.model,  test_cfg=config.get('test_cfg'))
    checkpoint = load_checkpoint(model, fstageckpt, map_location=device)
    model.CLASSES = 1
    model.cfg = config
    model.to(device)
    return model



if __name__ == "__main__":
    device='cuda:0'

    model_stage1 = {}
    for f in range(5):
        model_stage1[f] = loadModel (device, stage = 1, fold = f).eval()

    centers = glob("../data/test/*/")
    for c in centers:
        stats = [] # create a stats file too
        brokenstats = []

        print ("\n####### Processing center", c)
        pats = glob (os.path.join(c, "*/"))
        print ("Found", len(pats), "Patients.")
        pats = sorted(pats)
        for j, p in enumerate(pats):
            print ("\n###########\nProcessing patient", p)

            patID = os.path.basename(os.path.dirname(p))
            centername = c.split("/")[-2]
            anns = glob(f"../annotations/test/{centername}/{centername}_{patID}*.csv")

            if len(anns) > 1:
                print ("\tMore then one annotation found! This should not be possible.")
                raise Exception ("Too many annotations")

            # use the annotation only to determine if we should have not used the series
            if len(anns) == 1:
                ann = pd.read_csv(anns[0])
                try:
                    ann = ann.query("curFrame >= 0").iloc[0:1]
                except:
                    # should also never happen
                    raise Exception ("\Found file, but no annotation in there!")
                if "Deleted" in ann.iloc[0].keys():
                    if ann.iloc[0]["Deleted"] == True:
                        print ("\tAnnotation was deleted!")
                        continue
                # fix path.... stupid..
                ann["path"] = [x.replace("./data/", "../data/") for x in ann["path"]]
                # make evaluation 'easier'
                ann["true_slice"] = ann["curFrame"] # just copy from
            else:
                ann = pd.DataFrame.from_dict({"path": p}, orient = "index").T
            ann

            mrImg, spacing = loadMR(ann)
            ann["PX"] = spacing[0]
            ann["PY"] = spacing[1]

            # create montage
            fMontageImage = os.path.join(p, "Slices.png")
            montage = exportMontage (mrImg, fMontageImage)

            slicePreds = []
            valids = [0 for i in range(5)]
            for f in range(5):
                result_stage1 = inference_detector(model_stage1[f], fMontageImage)

                try:
                    box = mmrotate.core.bbox.obb2poly_np(result_stage1[0], version='le90')[0] # box with highest prob
                except:
                    box = [100, 0, 100, 100, 0, 100, 0, 0, 0.0]
                    valids[f] = 0
                    print ("Not found.")
                    continue
                valids[f] = 1

                if box is None:
                    pred_slice = 10 # corresponds to median of the gt slices
                else:
                    x0, y0, x1, y1, x2, y2, x3, y3, _= box
                    pred_bx = int(x0+x1+x2+x3)//4
                    pred_by = int(y0+y1+y2+y3)//4
                    pred_bx = pred_bx//(resize_size+border_size)
                    pred_by = pred_by//(resize_size+border_size)
                    pred_slice = pred_bx + pred_by*frame_size
                slicePreds.append(pred_slice)

            # take median
            print (slicePreds)
            print (valids)
            if np.sum(valids) < 3: # majority did not find anything
                pred_slice = sitk.GetArrayFromImage(mrImg).shape[0]//2
                print ("No preds at all!")
                brokenstats.append(ann)

            pred_slices = np.median(slicePreds)


            # extract pred_slice
            try:
                mrSlice = sitk.GetArrayFromImage(mrImg)[pred_slice, :, :]
            except:
                mrSlice = mrImg[pred_slice, :, :]
            neu = (mrSlice-np.min(mrSlice))/(np.max(mrSlice)-np.min(mrSlice)) * 255.0
            mrSlice = np.asarray(neu, dtype = np.uint8)
            image = np.stack((mrSlice[:,:],)*3, axis = -1)
            fSelectedSliceImage = os.path.join(p, "Selected_Slice.png")
            cv2.imwrite (fSelectedSliceImage, image)

            imageRGB = pseudoRGB (image[:,:,0])
            fSelectedSliceImage_RGB = os.path.join(p, "Selected_Slice_RGB.png")
            cv2.imwrite (fSelectedSliceImage_RGB, imageRGB)

            # add prediction to annotation and then to stats

            ann["predicted_slice"] = pred_slice
            stats.append(ann)
        df = pd.concat(stats).reset_index(drop = True)
        df.to_csv(os.path.join(c, "stats.csv"), index=False)
        print("c was", c)

        df = pd.DataFrame(brokenstats).reset_index(drop = True)
        df.to_csv(os.path.join(c, "broken_stats_stage1.csv"), index=False)
        print("c was", c)


#
