import argparse
import os

corTraAnnotation = False



def fixModel (cfg):
    print (cfg.model.type)
    if cfg.model.type == "RotatedRetinaNet":
        cfg.model.test_cfg.max_per_img = 3
        cfg.model.test_cfg.max_per_img = 3
    else:
        cfg.model.test_cfg.rpn.max_per_img = 3
        cfg.model.test_cfg.rcnn.max_per_img = 3


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


def fixTestData (cfg, args):
    # this is for testing only, i.e. FINAL testing
    testFile = "#"*50
    assert (args.fold == -1)
    assert (args.testset == args.testset)

    trainFile = None
    testFile = f"./annotations/{args.testset}_{args.margin}.json" # fake, we dont need this because there is no early stopping or whatever
    valFile = testFile

    cfg.train_dataloader.dataset.data_root = "."
    cfg.train_dataloader.dataset.ann_file = trainFile
    cfg.val_dataloader.dataset.data_root = "."
    cfg.val_dataloader.dataset.ann_file = valFile
    cfg.test_dataloader.dataset.data_root = "."
    cfg.test_dataloader.dataset.ann_file = testFile
    cfg.val_evaluator.ann_file = valFile
    cfg.test_evaluator.ann_file = testFile
    return cfg



def fixTrainData (cfg, args):
    # data
    root = "/data/data/prostata.roi/stage1/"
    trainFile = None
    if type(args.fold) == str:
        print ("Using given validation folder! Training is turned off!")
        trainAnnFile = None
        trainImgPrefix = None

        valAnnFile = f"{root}/{args.fold}/annotations"
        valImgPrefix = f"{root}/{args.fold}/images"

        testAnnFile = f"{root}/{args.fold}/annotations"
        testImgPrefix = f"{root}/{args.fold}/images"
    else:
        trainAnnFile = f"{root}/fold_{args.fold}/train/annotations"
        trainImgPrefix = f"{root}/fold_{args.fold}/train/images"

        valAnnFile = f"{root}/fold_{args.fold}/test/annotations"
        valImgPrefix = f"{root}/fold_{args.fold}/test/images"

        testAnnFile = f"{root}/fold_{args.fold}/test/annotations"
        testImgPrefix = f"{root}/fold_{args.fold}/test/images"

    cfg.data.train.ann_file = trainAnnFile
    cfg.data.train.img_prefix = trainImgPrefix
    cfg.data.val.ann_file = valAnnFile
    cfg.data.val.img_prefix = valImgPrefix
    cfg.data.test.ann_file = testAnnFile
    cfg.data.test.img_prefix = testImgPrefix
    # cfg.val_evaluator.ann_file = valFile
    # cfg.test_evaluator.ann_file = testFile
    return cfg



#
