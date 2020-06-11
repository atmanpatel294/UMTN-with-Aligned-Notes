# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
set -e -x

CODE=src
DATA=/datasets/tmp/dl4s/datasets/processed/midi_wav

EXP=cold_start_midi_small_model
export MASTER_PORT=29500

python ${CODE}/train.py \
    --data ${DATA}/Bach_Solo_Cello  \
           ${DATA}/Bach_Solo_Piano \
           ${DATA}/Beethoven_Accompanied_Violin \
           ${DATA}/Beethoven_String_Quartet  \
           ${DATA}/Johann_Sebastian_Bach \
           ${DATA}/Ludwig_van_Beethoven \
    --batch-size 30 \
    --lr-decay 0.995 \
    --epoch-len 1000 \
    --num-workers 0 \
    --lr 1e-3 \
    --seq-len 12000 \
    --d-lambda 1e-2 \
    --m-lambda 1 \
    --latent-d 64 \
    --layers 7 \
    --blocks 2 \
    --data-aug \
    --grad-clip 1 \
    --mode 4 \
    --num-decoders 5 \
    --d-channels 40 \
    --encoder-channels 32

#     --checkpoint checkpoints/pretrained_musicnet/bestmodel_0.pth \
