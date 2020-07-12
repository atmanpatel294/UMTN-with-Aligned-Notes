# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
set -e -x

CODE=src
# path to data folder should be according to multihot argument
# DATA=datasets/preprocessed
DATA=/datasets/tmp/dl4s/datasets/processed/midi_wav_0
# DATA=/datasets/tmp/dl4s/datasets/processed/midi_wav

EXP="final_warm_wav_midi_multihot_run3"
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
    --lr-decay 0.97 \
    --epoch-len 10 \
    --num-workers 0 \
    --lr 1e-3 \
    --seq-len 12000 \
    --d-lambda 1e-2 \
    --m-lambda 0.1 \
    --latent-d 64 \
    --layers 14 \
    --blocks 4 \
    --data-aug \
    --checkpoint checkpoints/pretrained_musicnet/bestmodel_0.pth \
    --grad-clip 1 \
    --mode 3 \
    --num-decoders 5 \
    --pretraining_epochs 25 \
    --multihot 0
    # --dict-size 70

        #    ${DATA}/Franz_Liszt \
        #    ${DATA}/Felix_Mendelssohn \
        #    ${DATA}/Franz_Schubert \
