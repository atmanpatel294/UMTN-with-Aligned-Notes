# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
set -e -x

CODE=src
DATA=datasets/preprocessed

EXP=musicnet_maestro_multi_decoders_with_durations_1
export MASTER_PORT=29500

python ${CODE}/train.py \
    --data ${DATA}/Franz_Liszt \
           ${DATA}/Cambini_Wind_Quintet \
           ${DATA}/Bach_Solo_Cello \
           ${DATA}/Felix_Mendelssohn  \
           ${DATA}/Franz_Schubert \
           ${DATA}/Johann_Sebastian_Bach \
           ${DATA}/Ludwig_van_Beethoven\
           ${DATA}/Beethoven_String_Quartet  \
    --batch-size 24 \
    --lr-decay 0.995 \
    --epoch-len 1000 \
    --num-workers 0 \
    --lr 1e-3 \
    --seq-len 12000 \
    --d-lambda 1e-2 \
    --expName ${EXP} \
    --latent-d 64 \
    --data-aug \
    --grad-clip 1 \
    --encoder-channels 32 \
    --blocks 2 \
    --layers 7 \
    --d-channels 40
    


       #     ${DATA}/Beethoven_Accompanied_Violin \
