# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import data
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help='Input directory')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Output directory')
    parser.add_argument('--norm-db', required=False, action='store_true')
    parser.add_argument('--data_type', type=str, required=True,
                        help='either wav or midi')
    parser.add_argument('--midi_type', type=str,
                        help='either chords or notes')

    args = parser.parse_args()
    print(args)
    if args.data_type == 'wav':
        dataset = data.EncodedFilesDataset(args.input)
        dataset.dump_to_folder(args.output, norm_db=args.norm_db)
    elif args.data_type=="midi":
        dataset = data.MidiCommonPreprocessor(args)
        dataset.dump_to_folder(args.output)
    else:
        print("\nERROR: Please enter correct mode from \nwav \nmidi")


if __name__ == '__main__':
    main()
