# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# DATE=`date +%d_%m_%Y`
# DATASET='maestro'
# Bach_Solo_Cello Beethoven_Accompanied_Violin Beethoven_String_Quartet Cambini_Wind_Quintet 
# Felix_Mendelssohn Franz_Liszt Franz_Schubert Johann_Sebastian_Bach Ludwig_van_Beethoven


INSTRUMENT='Bach_Solo_Cello'                #author/instrument folder
INPUT="datasets/preprocessed/${INSTRUMENT}" #input for sampling
CODE=src
OUTPUT="results/test_data/${INSTRUMENT}"    #output for sampling -> input for run_on_files
EXP="pretrained_musicnet"                   #location of decoder in checkpoints
DECODERS="0 1 2 3 4"                        #decoders to generate output for
BATCHSIZE=64

# echo "Sampling"
# python ${CODE}/data_samples.py --data ${INPUT} --output ${OUTPUT}  -n 5 --seq 80000

### python ${CODE}/data_samples.py --data-from-args checkpoints/$1/args.pth --output ${OUTPUT}  -n 4 --seq 80000

echo "Generating"
python ${CODE}/run_on_files.py --files ${OUTPUT} --batch-size $BATCHSIZE --py --checkpoint checkpoints/$EXP/d --output results/$EXP/$INSTRUMENT --decoders $DECODERS --decoder-id 0

# python ${CODE}/run_on_files.py --files ${OUTPUT} --batch-size 32 --py --checkpoint checkpoints/$EXP/d --output results/$EXP/$INSTRUMENT --decoders 0 --decoder-id 0 &
# python ${CODE}/run_on_files.py --files ${OUTPUT} --batch-size 32 --py --checkpoint checkpoints/$EXP/d --output results/$EXP/$INSTRUMENT --decoders 1 --decoder-id 1 &
# python ${CODE}/run_on_files.py --files ${OUTPUT} --batch-size 32 --py --checkpoint checkpoints/$EXP/d --output results/$EXP/$INSTRUMENT --decoders 2 --decoder-id 2 &
# python ${CODE}/run_on_files.py --files ${OUTPUT} --batch-size 32 --py --checkpoint checkpoints/$EXP/d --output results/$EXP/$INSTRUMENT --decoders 3 --decoder-id 3

INSTRUMENT='Beethoven_Accompanied_Violin'
OUTPUT="results/test_data/${INSTRUMENT}"
python ${CODE}/run_on_files.py --files ${OUTPUT} --batch-size $BATCHSIZE --py --checkpoint checkpoints/$EXP/d --output results/$EXP/$INSTRUMENT --decoders $DECODERS --decoder-id 0

INSTRUMENT='Beethoven_String_Quartet'
OUTPUT="results/test_data/${INSTRUMENT}"
python ${CODE}/run_on_files.py --files ${OUTPUT} --batch-size $BATCHSIZE --py --checkpoint checkpoints/$EXP/d --output results/$EXP/$INSTRUMENT --decoders $DECODERS --decoder-id 0

INSTRUMENT='Cambini_Wind_Quintet'
OUTPUT="results/test_data/${INSTRUMENT}"
python ${CODE}/run_on_files.py --files ${OUTPUT} --batch-size $BATCHSIZE --py --checkpoint checkpoints/$EXP/d --output results/$EXP/$INSTRUMENT --decoders $DECODERS --decoder-id 0

INSTRUMENT='Johann_Sebastian_Bach'
OUTPUT="results/test_data/${INSTRUMENT}"
python ${CODE}/run_on_files.py --files ${OUTPUT} --batch-size $BATCHSIZE --py --checkpoint checkpoints/$EXP/d --output results/$EXP/$INSTRUMENT --decoders $DECODERS --decoder-id 0
