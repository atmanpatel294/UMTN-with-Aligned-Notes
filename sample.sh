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


INSTRUMENT='Bach_Solo_Cello'
INPUT="datasets/preprocessed/${INSTRUMENT}"
CODE=src
OUTPUT="results/test_data/${INSTRUMENT}"
EXP="onehot"
DECODERS="0 5 6 7"

# echo "Sampling"
# python ${CODE}/data_samples.py --data ${INPUT} --output ${OUTPUT}  -n 5 --seq 80000

### python ${CODE}/data_samples.py --data-from-args checkpoints/$1/args.pth --output ${OUTPUT}  -n 4 --seq 80000

echo "Generating"
python ${CODE}/run_on_files.py --files ${OUTPUT} --batch-size 32 --py --checkpoint checkpoints/$EXP/decoder --output results/$EXP/test/$INSTRUMENT --decoders $DECODERS  --decoder-id 0

# INSTRUMENT='Beethoven_Accompanied_Violin'
# INPUT="datasets/preprocessed/${INSTRUMENT}"
# OUTPUT="results/test_data/${INSTRUMENT}"
# python ${CODE}/run_on_files.py --files ${OUTPUT} --batch-size 32 --py --checkpoint checkpoints/$EXP/decoder --output results/$EXP/$INSTRUMENT --decoders $DECODERS  

# INSTRUMENT='Beethoven_String_Quartet'
# INPUT="datasets/preprocessed/${INSTRUMENT}"
# OUTPUT="results/test_data/${INSTRUMENT}"
# python ${CODE}/run_on_files.py --files ${OUTPUT} --batch-size 32 --py --checkpoint checkpoints/$EXP/decoder --output results/$EXP/$INSTRUMENT --decoders $DECODERS  

# INSTRUMENT='Cambini_Wind_Quintet'
# INPUT="datasets/preprocessed/${INSTRUMENT}"
# OUTPUT="results/test_data/${INSTRUMENT}"
# python ${CODE}/run_on_files.py --files ${OUTPUT} --batch-size 32 --py --checkpoint checkpoints/$EXP/decoder --output results/$EXP/$INSTRUMENT --decoders $DECODERS  

# INSTRUMENT='Johann_Sebastian_Bach'
# INPUT="datasets/preprocessed/${INSTRUMENT}"
# OUTPUT="results/test_data/${INSTRUMENT}"
# python ${CODE}/run_on_files.py --files ${OUTPUT} --batch-size 32 --py --checkpoint checkpoints/$EXP/decoder --output results/$EXP/$INSTRUMENT --decoders $DECODERS  
