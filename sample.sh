# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# DATE=`date +%d_%m_%Y`
# DATASET='maestro'
INSTRUMENT='Franz_Liszt'
INPUT="datasets/preprocessed/${INSTRUMENT}"
CODE=src
OUTPUT="results/test_data/${INSTRUMENT}"
EXP="mode_3"
DECODERS="2 4"

echo "Sampling"
python ${CODE}/data_samples.py --data ${INPUT} --output ${OUTPUT}  -n 3 --seq 10000

### python ${CODE}/data_samples.py --data-from-args checkpoints/$1/args.pth --output ${OUTPUT}  -n 4 --seq 80000

echo "Generating"
# echo checkpoints/$EXP/something
python ${CODE}/run_on_files.py --files ${OUTPUT} --batch-size 4 --py --checkpoint checkpoints/$EXP/decoder --output results/$EXP/$INSTRUMENT --decoders $DECODERS  

#--output-next-to-orig
