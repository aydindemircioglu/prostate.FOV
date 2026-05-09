
from glob import glob
import numpy as np
import pandas as pd


def getAllAnnotations(basePath = ".."):
    aList = sorted(glob (f"{basePath}/annotations/train/*.csv"))
    print ("Have", len(aList), "annotations")
    invList = []
    vList = []
    for fann in aList:
        ann = pd.read_csv (fann)
        ann = ann.query('curFrame > 0').copy()
        if len(ann) == 0:
            print(ann)
            continue
        # also check for path
        vList.append(ann)
    print (f"Valid annotations: {len(vList)}")
    return vList



def getAllValidAnnotations():
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
    allAcc = np.array(sorted(list(set(allAnn["AccNr"].values))))
    return allAnn, allAcc


#
