
## conda  
follow conda things from webpage of mmdetection, and then activate it

$ source /home/aydin/.anaconda3/etc/profile.d/conda.sh
$ conda activate open-mmlab


## prepare data

./export_montage.py


## redet fix

the config for redet needs a special pretrained file.
download it from https://github.com/csuhan/ReDet/tree/master
and put it to /data/data/prostata.roi/pretrained
then fix ./configs/redet/redet_re50_refpn_1x_dota_le90.py
to point to that file


## train

$ CUDA_VISIBLE_DEVICES=0 python3 ./train.py configs/retinanet_r50.py --fold 0 --cfg-options data.samples_per_gpu=2     --gpu-ids 0 --seed 42 --deterministic --work-dir ./test_chkpt --LR 0.003


$ CUDA_VISIBLE_DEVICES=1 python3 ./test.py --fold 0 --cfg-options data.samples_per_gpu=2     --gpu-ids 0 --out out.pkl  --work-dir ./test_chkpt configs/retinanet_r50.py ./test_chkpt/latest.pth

CUDA_VISIBLE_DEVICES=1 python3 ./test.py --fold 0 --cfg-options data.samples_per_gpu=2     --gpu-ids 0 --out out.pkl  --work-dir /data/data/prostata.roi/checkpoints.montage/redet_r50/redet_r50_0_0.003/ configs/redet_r50.py /data/data/prostata.roi/checkpoints.montage/redet_r50/redet_r50_0_0.003/epoch_12.pth



then show_predictions.py




# on prostate-X on 2080ti

to export all data into mmdetect style in /data/data/prostate.roi/data.montage/extern/PROSTATEx/test:
  ./export_PROSTATEx.py

then do test using

CUDA_VISIBLE_DEVICES=1 python3 ./test_prostateX.py --fold 0 --cfg-options data.samples_per_gpu=2     --gpu-ids 0 --out out_prostate_X.pkl  --work-dir /data/data/prostata.roi/checkpoints.montage/redet_r50/redet_r50_0_0.003/ configs/redet_r50.py /data/data/prostata.roi/checkpoints.montage/redet_r50/redet_r50_0_0.003/epoch_12.pth


then visualize
show_preditions_prostateX.py


# todo
./test/pat_1/dcm
erzeugt dann ./test/pat_1/slice_selection.png und ./annotated_slice.png
montage braucht aprameter applyRandomOffset = Yes/No, waehrend test ist es dann 0.
