#!/bin/bash

python3 ./standalone_stage2.py -f 0
python3 ./standalone_stage2.py -f 1
python3 ./standalone_stage2.py -f 2
python3 ./standalone_stage2.py -f 3
python3 ./standalone_stage2.py -f 4
python3 ./standalone_stage2_pred.py

#
