#!/usr/bin/python3

import os
import pandas as pd
import numpy as np
import cv2
import sys
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
BACK = ord('b')  # MAC 167, ME 96
ESC = 27



clipFactor = 2
clipLimit = 2

wmin, wmax = None, None
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
        self.parser.add_argument('--split', type=str, default=None, help='split to annotate.')
        self.parser.add_argument('--rater', type=str, default=None, help='rater to annotate.')
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




if  __name__ == '__main__':
    opt = BaseOptions().parse()
    if opt.split is None or opt.rater is None:
        raise Exception ("Need rater and split!")
    print (opt.split)
    print (opt.rater)
    cohort = opt.split

    os.makedirs("./results/", exist_ok = True)
    try:
        df = pd.read_csv(f"results/{cohort}_eval.csv")
    except Exception as e:
        # does not exist, so take 'fresh' one
        print (cohort)
        df = pd.read_csv(f"../data/test/{cohort}/stats.csv")
        df = df.reset_index(drop = True)
        # wtf
        for z in range(len(df)):
            srcimg = df.iloc[z]["path"]
            print (df.iloc[z])
            if "uke" == cohort:
                # this is uke, some have "./MR/<accNR>/t2.."
                srcimg = "uke/"+srcimg.split("/")[2]
            else:
                for j, p in enumerate(srcimg.split("/")):
                    if p == cohort:
                        break
                srcimg = srcimg.split("/")[j] + "/" + srcimg.split("/")[j+1]
            srcimg = "../data/test/"+os.path.join(srcimg, "Final_Slice_Prediction.png")
            print (srcimg)
            if os.path.exists(srcimg) == False:
                raise Exception ("Does not exist?", srcimg)
            df.at[z,"path"] = srcimg
        df = df[["path"]]


    # check that rater is there
    try:
        z = df.iloc[0][f"Stage1_{opt.rater}"]
    except:
        df[f"Stage1_{opt.rater}"] = 0
        df[f"Stage2_{opt.rater}"] = 0
    curMR = 0
    maxMR = len(df)

    while True:
        # ensure that index of current MR exists
        if curMR < 0:
            curMR = 0
        if curMR > maxMR - 1:
            curMR = maxMR - 1

        current = 0
        cv2.namedWindow ("MR", cv2.WINDOW_GUI_EXPANDED)
        lastOne = 0
        gotoNext = 0
        k = 0
        while 1 == 1:
            srcimg = df.iloc[curMR]["path"]
            srcimg = cv2.imread(srcimg)

            displayImg = srcimg.copy()
            if df.at[curMR,f"Stage1_{opt.rater}"] == 1:
                cv2.putText(displayImg, "Wrong slice", (0,displayImg.shape[0]//4), font, 1.4, (0, 0, 255), 3)
            if df.at[curMR,f"Stage2_{opt.rater}"] == 1:
                cv2.putText(displayImg, "Bad annotation", (0,3*displayImg.shape[0]//4), font, 1.4, (0, 0, 255), 3)
            cv2.imshow("MR", displayImg)
            while k != ESC and k != SKIP and k != BACK and k != K_9:
                k = cv2.waitKey(30) & 0xFF
                #print(k)
                if k == 255:
                    continue


                if k == K_q:
                    df.at[curMR, f"Stage1_{opt.rater}"] = 1 - df.at[curMR, f"Stage1_{opt.rater}"]
                    print ("Now:", df.at[curMR, f"Stage1_{opt.rater}"] )

                if k == K_e:
                    df.at[curMR, f"Stage2_{opt.rater}"] = 1 - df.at[curMR, f"Stage2_{opt.rater}"]
                    print ("Now:", df.at[curMR, f"Stage2_{opt.rater}"] )


                displayImg = srcimg.copy()
                if df.at[curMR,f"Stage1_{opt.rater}"] == 1:
                    cv2.putText(displayImg, "Wrong slice", (0,displayImg.shape[0]//4), font, 1.4, (0, 0, 255), 3)
                if df.at[curMR,f"Stage2_{opt.rater}"] == 1:
                    cv2.putText(displayImg, "Bad annotation", (0,3*displayImg.shape[0]//4), font, 1.4, (0, 0, 255), 3)
                cv2.imshow("MR", displayImg)



                if k == BACK:
                    gotoNext = -1
                    break

                if k == SKIP:
                    gotoNext = 1
                    break

                if k == K_7:
                    gotoNext = -25
                    break

                if k == K_9:
                    gotoNext = 25
                    break

                if k == 27:
                    print ("Stopping.")
                    df.to_csv(f"results/{cohort}_eval.csv", index = False)
                    exit(-1)

            if gotoNext != 0:
                break
        curMR = curMR + gotoNext

    df.to_csv(f"results/{cohort}_eval.csv", index = False)
    cv2.destroyAllWindows()

#
