# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
set -e -x

CODE=src
# DATA=datasets/preprocessed
DATA=/datasets/tmp/dl4s/datasets/processed/midi_wav_0

EXP="final_warm_wav_midi_mode1"
export MASTER_PORT=29500

python ${CODE}/train.py \
    --expName ${EXP} \
    --data ${DATA}/Bach_Solo_Cello \
           ${DATA}/Bach_Solo_Piano \
           ${DATA}/Beethoven_Accompanied_Violin \
           ${DATA}/Beethoven_String_Quartet \
           ${DATA}/Johann_Sebastian_Bach \
           ${DATA}/Ludwig_van_Beethoven \
    --batch-size 12 \
    --lr-decay 0.95 \
    --epoch-len 1000 \
    --num-workers 0 \
    --lr 1e-3 \
    --seq-len 12000 \
    --d-lambda 1e-2 \
    --m-lambda 1 \
    --latent-d 64 \
    --layers 14 \
    --blocks 4 \
    --data-aug \
    --checkpoint checkpoints/pretrained_musicnet/bestmodel_0.pth \
    --grad-clip 1 \
    --mode 1 \
    --num-decoders 5 \
    --pretraining_epochs 20
        #    ${DATA}/Franz_Liszt \
        #    ${DATA}/Felix_Mendelssohn \
        #    ${DATA}/Franz_Schubert \
