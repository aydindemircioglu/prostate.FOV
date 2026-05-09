import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from PIL import Image, ImageEnhance

import sys
sys.path.append("..")
from utils_path import *


noise_levels = [0.1, 0.3, 0.5, 1.0]
contrast_levels = [0.1, 0.3, 0.5, 1.0]


centers = glob("../data/test/*/")
for c in centers:
    center = os.path.basename(os.path.dirname(c))
    # no thanks
    if "sensitivity" in c:
        continue
    print ("\n####### Processing center", c)
    try:
        print ("Reading stats..")
        stats = pd.read_csv(f'{c}/stats.csv')
    except:
        print (f'Whats up with {c}/stats.csv?')
        print ('First execute stage 1!')
        continue

    if "angle_diff_cor_abs" not in stats.keys():
        print ("Either no ground truth or stage 2 was not executed!")
        continue
    print ("Preparing center ", c)

    center_name = c.split("/")[3]
    outputPath = f"../data/test/{center_name}_sensitivity"
    recreatePath (outputPath)

    # for each set pick 30 images
    np.random.seed(42)
    subset = stats.sample(n=min(len(stats), 30))
    allstats = []

    for j, (idx, row) in enumerate(subset.iterrows()):
        # load gt slice image
        ctpath = row["path"]
        accID = os.path.basename(os.path.dirname(ctpath))
        gtimg = cv2.imread(f"{ctpath}/Selected_Slice_RGB.png")

        # noise sensitivity
        noise_levels = [0, 1,2,4,8,16]
        for n in noise_levels:
            if n > 0:
                noise = np.random.normal(0, n, gtimg.shape)
                nimg = cv2.add(gtimg.astype(np.float64), noise)
                nimg = np.clip(nimg, 0, 255).astype(np.uint8)
            else:
                nimg = gtimg.copy()
            nimgPath = f"{outputPath}/{center}_{accID}_noise_{n}"
            recreatePath (nimgPath)
            cv2.imwrite(f"{nimgPath}/Selected_Slice_RGB.png", nimg)
            crow = row.copy()
            crow["noise"] = n
            crow["path"] = nimgPath
            allstats.append(crow)


        contrast_levels = [0, 1, 2, 4, 8, 16]
        for cl in contrast_levels:
            if cl > 0:
                pil_img = Image.fromarray(gtimg)
                enhancer = ImageEnhance.Contrast(pil_img)
                enhanced_img = enhancer.enhance((1+cl/16))
                nimg = np.array(enhanced_img)
            else:
                nimg = gtimg.copy()

            nimgPath = f"{outputPath}/{center}_{accID}_contrast_{cl}"
            recreatePath (nimgPath)
            cv2.imwrite(f"{nimgPath}/Selected_Slice_RGB.png", nimg)
            crow = row.copy()
            crow["contrast"] = cl
            crow["path"] = nimgPath
            allstats.append(crow)

    allstats = pd.DataFrame(allstats)
    allstats.to_csv(f'{outputPath}/stats.csv', index = False)

#
