#!/bin/bash

rm -rf ../data/stage1/checkpoints
mkdir -p ../data/stage1/checkpoints

#python3 ./prepareTrainData.py

CUDA_VISIBLE_DEVICES=0
for cfg in  redet_r50 rotated_retinanet_r50
do
  for f in 0 1 2 3 4
  do
    for lr in 0.08 0.04 0.02 0.008 0.004
    do
      wd=../data/stage1/checkpoints/$cfg/${cfg}_${f}_${lr}
      fpkl=../data/stage1/checkpoints/$cfg/${cfg}_${f}_${lr}.pkl
      python3 ./train.py configs/$cfg.py --fold $f --cfg-options data.samples_per_gpu=8     --gpu-ids 0 --seed 42 --deterministic --work-dir $wd --LR $lr
      python3 ./test.py --fold $f --cfg-options data.samples_per_gpu=8     --gpu-ids 0 --out $fpkl  --work-dir $wd configs/$cfg.py $wd/latest.pth
    done
  done
done

python3 ./modelSelection.py

#
