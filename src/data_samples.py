# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import data
import argparse
from pathlib import Path
import tqdm
import random
import pdb

from utils import inv_mu_law, save_audio
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, nargs='*',
                        help='Path to data dir')
    parser.add_argument('--data-from-args', type=Path,
                        help='Path to args.pth')
    parser.add_argument('--output', '-o', type=Path,
                        help='Output path')
    parser.add_argument('-n', type=int,
                        help='Num samples to make')
    parser.add_argument('--seq-len', type=int, default=80000)

    args = parser.parse_args()

    if args.data:
        dataset_name = args.data[0].parts[-1]
        print("creating dataset from ", dataset_name)
        datasets = data.H5Dataset(args.data[0] / 'test', args.seq_len, 'wav', mode=1)

    else:
        print("Please provide --data argument")
        return
    
    for dataset_id, dataset in enumerate(datasets):
        if dataset_id>=args.n:
            break
        wav_data,_,_ = dataset
        wav_data = inv_mu_law(wav_data)
        file_name = "{}/{}.wav".format(args.output, dataset_id)
        save_audio(wav_data, Path(file_name), rate=data.EncodedFilesDataset.WAV_FREQ)



if __name__ == '__main__':
    main()
