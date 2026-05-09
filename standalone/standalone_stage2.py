#!/usr/bin/python3
import os
import numpy as np
import cv2
from glob import glob
import pandas as pd
import sys
from joblib import dump
import argparse

import mmcv
from mmengine.runner import load_checkpoint

from mmdet.apis import inference_detector
from mmrotate.models import build_detector
import mmrotate



class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('-f', type = str, default = None, help = 'fold to predict.')

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


def loadModel (device = "cuda:0", stage = None, fold = None):
    fcfg = "models/stage2_cfg.py"
    fstageckpt = f"./models/stage{stage}_f{fold}.pth"

    config = mmcv.Config.fromfile(fcfg)
    config.model.pretrained = None

    def fixModel (cfg):
        print (cfg.model.type)
        if cfg.model.type == "CascadeRCNN":
            for k in range(len(cfg.model.roi_head.bbox_head)):
                cfg.model.roi_head.bbox_head[k]["num_classes"] = 2
        elif cfg.model.type == "RotatedRetinaNet":
            cfg.model.bbox_head["num_classes"] = 2
        elif cfg.model.type == "ReDet":
            for k in range(len(cfg.model.roi_head.bbox_head)):
                cfg.model.roi_head.bbox_head[k]["num_classes"] = 2
        elif cfg.model.type == "RoITransformer":
            for k in range(len(cfg.model.roi_head.bbox_head)):
                cfg.model.roi_head.bbox_head[k]["num_classes"] = 2
        else:
            raise Exception ("Unknown model, cannot adapt number of classes.")
        return cfg
    config = fixModel(config)

    model = build_detector(config.model,  test_cfg=config.get('test_cfg'))
    checkpoint = load_checkpoint(model, fstageckpt, map_location=device)
    model.CLASSES = 2
    model.cfg = config
    model.to(device)
    return model


if __name__ == "__main__":
    device='cuda:0'

    opt = BaseOptions().parse()
    if opt.f is None:
        raise Exception ("Need fold!")
    centers = glob("../data/test/*/")
    print ("Predicting on centers", centers)

    f = opt.f
    try:
        f = int(f)
        print("Using fold:", f)
    except ValueError:
        print("Invalid fold number")
        exit(-1)

    model_stage2 = loadModel (device, stage = 2, fold = f).eval()
    results = {}

    for c in centers:
        print ("\n####### Processing center", c)
        try:
            print ("Reading stats..")
            stats = pd.read_csv(f'{c}/stats.csv')
        except:
            print (f'Whats up with {c}/stats.csv')
            print ('First execute stage 1!')
            exit(-1)

        # reset stags
        stats[f"stage2_{f}_cor"] = ''
        stats[f"stage2_{f}_tra"] = ''

        basePath = c

        print ("Found", len(stats), "Patients.")
        for j, (idx, row) in enumerate(stats.iterrows()):
            p = row["path"]
            print ("\n\n\n\n###########\nProcessing patient", p)
            fSelectedSliceImage_RGB = os.path.join(p, "Selected_Slice_RGB.png")
            if os.path.exists(fSelectedSliceImage_RGB) == False:
                raise Exception ("\tNot found in stats. Maybe no annotation.")

            print ("Processing")

            result_stage2 = {"cor": [], "tra":[]}
            curResult = inference_detector(model_stage2, fSelectedSliceImage_RGB)

            try:
                cor = mmrotate.core.bbox.obb2poly_np(curResult[0], version='le90')[0] # box with highest prob
                tra = mmrotate.core.bbox.obb2poly_np(curResult[1], version='le90')[0] # box with highest prob
                print ("PROBABILITY COR:", str(np.round(100*cor[-1])))
                print ("PROBABILITY TRA:", str(np.round(100*tra[-1])))
                result_stage2["cor"].append(cor)
                result_stage2["tra"].append(tra)
            except:
                message = f"Patient {p} stage 2 fold {f} prediction failed!"
                print (message)
            stats.at[idx, f"stage2_{f}_cor"] = repr(cor)
            stats.at[idx, f"stage2_{f}_tra"] = repr(tra)
        stats.to_csv(f'{c}/stats.csv', index = False)
#
