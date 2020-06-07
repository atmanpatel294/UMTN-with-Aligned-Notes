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

EXP=musicnet_maestro_multiple_decoders
export MASTER_PORT=29500

python ${CODE}/train.py \
    --data ${DATA}/Bach_Solo_Cello \
           ${DATA}/Cambini_Wind_Quintet \
           ${DATA}/Beethoven_String_Quartet \
           ${DATA}/Franz_Liszt \
           ${DATA}/Felix_Mendelssohn \
           ${DATA}/Franz_Schubert \
           ${DATA}/Johann_Sebastian_Bach \
           ${DATA}/Ludwig_van_Beethoven\
    --num-decoders 4 \
    --batch-size 30 \
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
    


#     --checkpoint checkpoints/test/bestmodel_0.pth \
       #     ${DATA}/Beethoven_Accompanied_Violin \
