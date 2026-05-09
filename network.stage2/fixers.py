import argparse
import os


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


    # if args.testset is None:
    #     if args.fold < 0:
    #         testFile = f"./annotations/train_{args.margin}.json"
    #     else:
    #         testFile = f"./annotations/train_val_fold_{args.fold}_{args.margin}.json"
    # else:
    #     #???
    #     testFile = "./annotations/" + str(args.testset) + "_" + str(args.testset) + ".json" # har har, i am a pirate
    #
    # cfg.test_dataloader.dataset.data_root = "."
    # cfg.test_dataloader.dataset.ann_file = testFile
    # return cfg
    #
    # if cfg.model.type == "CascadeRCNN" or cfg.model.type == "VFNet" or cfg.model.type == "RetinaNet"  or \
    #            cfg.model.type == "CornerNet" or cfg.model.type == "FSAF":
    #     cfg.data.test.ann_file = testFile
    # elif cfg.model.type == "YOLOX":
    #     # what is this?... possibly the same
    #     cfg.data.test.ann_file = testFile
    #     cfg.evaluation.interval = 5
    #
    # else:
    #     raise Exception ("Unknown model, cannot adapt datasets.")
    #
    return cfg



def fixTrainData (cfg, args):
    # data
    root = "/data/data/prostata.roi/stage2/"
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
