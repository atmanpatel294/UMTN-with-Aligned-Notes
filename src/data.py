# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import random
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import librosa
import numpy as np
import torch
import torch.utils.data as data
import tqdm
from scipy.io import wavfile
import pickle 
from music21 import note, chord, converter
from multiprocessing import Pool

import utils
from utils import mu_law

import os

import pdb

logger = utils.setup_logger(__name__, 'data.log')


def random_of_length(seq, length):
    limit = seq.size(0) - length
    if length < 1:
        # logging.warning("%d %s" % (length, path))
        return None

    start = random.randint(0, limit)
    end = start + length
    return seq[start: end]


class MidiCommonPreprocessor():
    """
    Uses music21 to read midi files and save them as both
    one-hot and multi-hot
    """
    FILE_TYPES = ['midi', 'mid']
    SEQ_LEN = 1.0
    
    def __init__(self, args):
        self.args = args
        print("\n--- arguments ---\n", args)
        self.path = Path(args.input)
        self.outpath = Path(args.output)
        ## same for notes and chords
        self.noteToInt, self.intToNote = {'SOS':0, 'EOS':1}, {0:'SOS', 1:'EOS'}
        self.file_paths = self.filter_paths(self.path.glob('**/*'), self.FILE_TYPES)
        print("\n-- Number of files found : {} --\n".format(len(self.file_paths)))
        # print("file_paths", self.file_paths)

    def extract_chords_duration_time_from_song(self, song, folder):
        score = converter.parse(song)
        score_chordify = score.chordify()
        chords,durations,start_times = [],[],[]
        unique_counter = len(self.noteToInt) #0 and 1 will be used for SOS and EOS

        for element in score_chordify:
            if isinstance(element, note.Note):
                note_ = str(element.pitch)
                if folder=="/train/":
                    if note_ not in self.noteToInt:
                        self.noteToInt[note_] = unique_counter
                        self.intToNote[unique_counter] = note_
                        unique_counter += 1
                chords.append(note_)
                durations.append(round(float(element.duration.quarterLength),3))
                start_times.append(float(element.offset))
            elif isinstance(element, chord.Chord):
                # Music21: Generally using .pitches or iterating over the chord is the best 
                # way to work with the components of a chord
                if self.args.midi_type == 'notes':
                    for note_ in element.pitches:
                        note_ = str(note_)
                        if folder=="/train/":
                            if note_ not in self.noteToInt:
                                self.noteToInt[note_] = unique_counter
                                self.intToNote[unique_counter] = note_
                                unique_counter += 1
                    curr_chord = [str(n) for n in element.pitches]
                elif self.args.midi_type == 'chords':
                    curr_chord = '.'.join(str(n) for n in sorted(element.pitches))
                    if folder=='/train/':
                        if curr_chord not in self.noteToInt:
                            self.noteToInt[curr_chord] = unique_counter
                            self.intToNote[unique_counter] = curr_chord
                            unique_counter += 1
                else:
                    print("ERROR: Encoding value in args is incorrect!")
                chords.append(curr_chord)
                durations.append(round(float(element.duration.quarterLength),3))
                start_times.append(float(element.offset))
        
        return [start_times,chords,durations]
    
    def chord_to_hot(self, chord):
        ret = [0]*len(self.noteToInt)
        if isinstance(chord, str):
            if chord in self.noteToInt:
                ret[self.noteToInt[chord]] = 1
        if isinstance(chord, list):
            for note in chord:
                if note in self.noteToInt:
                    ret[self.noteToInt[note]] = 1
        return ret
        

    def save_dict_as_txt(self, D):
        print("Saving the vocab at ", str(self.outpath))
        #input D: intToChord
        note_list = []
        for i in range(len(D)):
            note_list.append(D[i])
        filename = self.outpath / "vocab.txt"
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            for item in note_list:
                f.write("%s\n" % item)


    def dump_to_folder(self, output:Path):
        folders = ['/train/','/test/','/val/']
        for folder in folders:
            print("Processing Folder ", folder)
            curr_file_paths = [f for f in self.file_paths if folder in str(f)]
            num_songs = len(curr_file_paths)

            original_scores = []

            # with Pool(16) as p:
            #     processed_song_list = p.map(self.extract_chords_duration_time_from_song, songList, folder)

            processed_song_list = []
            for song in tqdm.tqdm(curr_file_paths):
                processed_song_list.append(self.extract_chords_duration_time_from_song(song, folder))

            if folder=="/train/":
                print("----------------  Indexing  ----------------")
                print("number of unique notes/chords : ", len(self.noteToInt))
                # pickle.dump(self.noteToInt, open(str(self.path) + "/noteToInt.pkl", "wb"))
                # pickle.dump(self.intToNote, open(str(self.path) + "/intToNote.pkl", "wb"))
                self.save_dict_as_txt(self.intToNote)
                print("---------------- dict saved ----------------\n\n")
            
            print("Converting chord to hot")
            # for i,song in enumerate(processed_song_list):
            #     for j,chord in enumerate(song[1]):
            #         ## song = [starttimes, chords, durations]
            #         ## changes in song are not reflected in processed_song_list
            #         song[1][j] = self.chord_to_hot(chord) 
            #     processed_song_list[i] = song

            # for i,song in enumerate(curr_file_paths):
            #     output_file_path = output / song.relative_to(self.path).with_suffix('.pkl')
            #     output_file_path.parent.mkdir(parents=True, exist_ok=True)
            #     pickle.dump(np.array(processed_song_list[i]), open(str(output_file_path), "wb"))

            for fname,song in tqdm.tqdm(zip(curr_file_paths, processed_song_list)):
                for j,chord in enumerate(song[1]):
                    song[1][j] = self.chord_to_hot(chord) 
                
                output_file_path = output / fname.relative_to(self.path).with_suffix('.pkl')
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                pickle.dump(np.array(song), open(str(output_file_path), "wb"))




    
    @staticmethod
    def filter_paths(haystack, file_types):
        return [f for f in haystack
                if (f.is_file()
                    and any(f.name.endswith(suffix) for suffix in file_types)
                    and '__MACOSX' not in f.parts)]

# class MidiMultiHotFileDataset():
#     """
#     Uses music21 to read midi files
#     """
#     FILE_TYPES = ['midi', 'mid']
#     SEQ_LEN = 1.0
    
#     def __init__(self, top, file_type=None):
#         self.path = Path(top)
#         self.noteToInt, self.intToNote = {}, {}
        
#         self.file_types = [file_type] if file_type else self.FILE_TYPES
#         self.file_paths = self.filter_paths(self.path.glob('**/*'), self.file_types)
#         print("file_paths", self.file_paths)
        
#     def extract_chords_duration_time_from_song(self, song, folder):
#         score = converter.parse(song)
#         score_chordify = score.chordify()
#         chords,durations,times = [],[],[]
#         val_counter = 0
#         for element in score_chordify:
#             if isinstance(element, note.Note):
#                 note_ = str(element.pitch)
#                 if folder=="/train/":
#                     if note_ not in self.noteToInt:
#                         # print(note_)
#                         self.noteToInt[note_] = val_counter
#                         self.intToNote[val_counter] = note_
#                         val_counter += 1
#                 chords.append(note_)
#                 durations.append(round(float(element.duration.quarterLength),3))
#                 times.append(float(element.offset))
#             elif isinstance(element, chord.Chord):
#                 # Music21: Generally using .pitches or iterating over the chord is the best 
#                 # way to work with the components of a chord
#                 for note_ in element.pitches:
#                     note_ = str(note_)
#                     if folder=="/train/":
#                         if note_ not in self.noteToInt:
#                             # print(note_)
#                             self.noteToInt[note_] = val_counter
#                             self.intToNote[val_counter] = note_
#                             val_counter += 1
#                 chords.append([str(n) for n in element.pitches])
#                 durations.append(round(float(element.duration.quarterLength),3))
#                 times.append(float(element.offset))
        
#         return [times,chords,durations]
    
#     def chord_to_hot(self, chord):
#         ret = [0]*len(self.noteToInt)  #np.zeros(len(noteToInt))
#         # print(type(chord),chord)
#         if isinstance(chord, str):
#             if chord in self.noteToInt:
#                 ret[self.noteToInt[chord]] = 1
#         if isinstance(chord, list):
#             for note in chord:
#                 if note in self.noteToInt:
#                     ret[self.noteToInt[note]] = 1
#         # print(type(ret), ret)
#         return ret
        

#     def dump_to_folder(self, output:Path):
#         folders = ['/train/','/test/','/val/']
#         # intToNote, noteToInt = {},{}
#         for folder in folders:
#             songList = [f for f in self.file_paths if folder in str(f)]
#             num_songs = len(songList)

#             original_scores = []

#             # with Pool(16) as p:
#             #     processed_song_list = p.map(self.extract_chords_duration_time_from_song, songList, folder)

#             processed_song_list = []
#             for song in songList:
#                 processed_song_list.append(self.extract_chords_duration_time_from_song(song, folder))

#             if folder=="/train/":
#                 # noteToInt, intToNote = _noteToInt, _intToNote
#                 ## assuming all the notes are convered in the train dataset
#                 print("----------------  Indexing  ----------------")
#                 print("number of unique notes : ", len(self.noteToInt))
#                 pickle.dump(self.noteToInt, open(str(self.path) + "/noteToInt.pkl", "wb"))
#                 pickle.dump(self.intToNote, open(str(self.path) + "/intToNote.pkl", "wb"))
#                 print(self.noteToInt)
#                 print("dict saved\n\n")

#             # indexedChords = [[chordToInt[c[0]] if c[0] in chordToInt else 0 for c in f] for f in processed_song_list]
#             # onehot_song_list = []
#             for i,song in enumerate(processed_song_list):
#                 for j,chord in enumerate(song[1]):
#                     song[1][j] = self.chord_to_hot(chord) # changes in song are not reflected in processed_song_list
#                 processed_song_list[i] = song
            


#             # combined = list(zip(originalTiming, indexedChords, originalDurations))
#             #save file
#             # print(processed_song_list)
#             for i,song in enumerate(songList):
#                 output_file_path = output / song.relative_to(self.path).with_suffix('.pkl')
#                 output_file_path.parent.mkdir(parents=True, exist_ok=True)
#                 pickle.dump(np.array(processed_song_list[i]), open(str(output_file_path), "wb"))
    
#     @staticmethod
#     def filter_paths(haystack, file_types):
#         return [f for f in haystack
#                 if (f.is_file()
#                     and any(f.name.endswith(suffix) for suffix in file_types)
#                     and '__MACOSX' not in f.parts)]


# class MidiFileDataset():
#     """
#     Uses music21 to read midi files
#     """
#     FILE_TYPES = ['midi', 'mid']
#     SEQ_LEN = 1.0
    
#     def __init__(self, top, file_type=None):
#         self.path = Path(top)
#         self.file_types = [file_type] if file_type else self.FILE_TYPES
#         self.file_paths = self.filter_paths(self.path.glob('**/*'), self.file_types)
#         print("file_paths", self.file_paths)
        
#     def extract_chords_duration_time_from_song(self, song):

#         score = converter.parse(song)
#         score_chordify = score.chordify()
#         chords_duration_time = []
#         chords,durations,times = [],[],[]
#         for element in score_chordify:
#             if isinstance(element, note.Note):
#                 chords.append(element.pitch)
#                 durations.append(round(float(element.duration.quarterLength),3))
#                 times.append(float(element.offset))
#             elif isinstance(element, chord.Chord):
#                 chords.append('.'.join(str(n) for n in sorted(element.pitches)))
#                 durations.append(round(float(element.duration.quarterLength),3))
#                 times.append(float(element.offset))
#         return [times,chords,durations]
        
        
#     def dump_to_folder(self, output:Path):
#         #remove old folders
#         folders = ['/train/','/test/','/val/']
#         # for folder in folders:
#         #     if os.path.exists(target_dir+folder):
#         #         shutil.rmtree(target_dir+folder)
#         #     os.makedirs(target_dir+folder)

#         uniqueChords,uniqueDurations = [],[]
#         intToChord, intToDuration, chordToInt, durationToInt = {},{},{},{}
#         for folder in folders:
#             # songList = [f for f in os.listdir(data_dir+folder) if not f.startswith('.')]
#             songList = [f for f in self.file_paths if folder in str(f)]
            
#             num_songs = len(songList)

#             original_scores = []

#             # multithreading
#             with Pool(16) as p:
#                 processed_song_list = p.map(self.extract_chords_duration_time_from_song, songList)
            
#             if folder=="/train/":
#                 print("----------------  Indexing  ----------------")
#                 uniqueChords = np.unique([i for s in processed_song_list for i in s[1]])
#                 chordToInt = dict(zip(uniqueChords, list(range(1, len(uniqueChords)+1))))

#                 # Map unique durations to integers
#                 # uniqueDurations = np.unique([i for s in processed_song_list for i in s[2]])

#                 intToChord = {i: c for c, i in chordToInt.items()}

#                 print("number of unique chords : ", len(uniqueChords))
#                 # print("number of unique durations", len(uniqueDurations))

#                 pickle.dump(chordToInt, open(str(self.path) + "/chordToInt.pkl", "wb"))
#                 pickle.dump(intToChord, open(str(self.path) + "/intToChord.pkl", "wb"))

#             # indexedChords = [[chordToInt[c[0]] if c[0] in chordToInt else 0 for c in f] for f in processed_song_list]
#             for song in processed_song_list:
#                 for i,chord in enumerate(song[1]):
#                     if chord in chordToInt:
#                         song[1][i] = chordToInt[chord]
#                     else:
#                         song[1][i] = 0
            
#             # indexedDurations = originalDurations #[[durationToInt[c] for c in f] for f in originalDurations]

#             # combined = list(zip(originalTiming, indexedChords, originalDurations))
#             # save files
#             for i,song in enumerate(songList):
#                 output_file_path = output / song.relative_to(self.path).with_suffix('.pkl')
#                 output_file_path.parent.mkdir(parents=True, exist_ok=True)
#                 pickle.dump( np.array(processed_song_list[i]), open(str(output_file_path), "wb"))
    
#     @staticmethod
#     def filter_paths(haystack, file_types):
#         return [f for f in haystack
#                 if (f.is_file()
#                     and any(f.name.endswith(suffix) for suffix in file_types)
#                     and '__MACOSX' not in f.parts)]



class EncodedFilesDataset(data.Dataset):
    """
    Uses ffmpeg to read a random short segment from the middle of an encoded file
    """
    FILE_TYPES = ['mp3', 'ape', 'm4a', 'flac', 'mkv', 'wav']
    WAV_FREQ = 16000
    INPUT_FREQ = 44100
    FFT_SZ = 2048
    WINLEN = FFT_SZ - 1
    HOP_SZ = 80

    def __init__(self, top, seq_len=None, file_type=None, epoch_len=10000):
        self.path = Path(top)
        self.seq_len = seq_len
        self.file_types = [file_type] if file_type else self.FILE_TYPES
        self.file_paths = self.filter_paths(self.path.glob('**/*'), self.file_types)
        self.epoch_len = epoch_len

    @staticmethod
    def filter_paths(haystack, file_types):
        return [f for f in haystack
                if (f.is_file()
                    and any(f.name.endswith(suffix) for suffix in file_types)
                    and '__MACOSX' not in f.parts)]

    def _random_file(self):
        # return np.random.choice(self.file_paths, p=self.probs)
        return random.choice(self.file_paths)

    @staticmethod
    def _file_length(file_path):
        output = subprocess.run(['ffprobe',
                                 '-show_entries', 'format=duration',
                                 '-v', 'quiet',
                                 '-print_format', 'compact=print_section=0:nokey=1:escape=csv',
                                 str(file_path)],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE).stdout
        duration = float(output)

        return duration

    def _file_slice(self, file_path, start_time):
        length_sec = self.seq_len / self.WAV_FREQ
        length_sec += .01  # just in case
        with NamedTemporaryFile() as output_file:
            output = subprocess.run(['ffmpeg',
                                     '-v', 'quiet',
                                     '-y',  # overwrite
                                     '-ss', str(start_time),
                                     '-i', str(file_path),
                                     '-t', str(length_sec),
                                     '-f', 'wav',
                                     # '-af', 'dynaudnorm',
                                     '-ar', str(self.WAV_FREQ),  # audio rate
                                     '-ac', '1',  # audio channels
                                     output_file.name
                                     ],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE).stdout
            rate, wav_data = wavfile.read(output_file)
            assert wav_data.dtype == np.int16
            wav = wav_data[:self.seq_len].astype('float')

            return wav

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, _):
        wav = self.random_file_slice()
        return torch.FloatTensor(wav)

    def random_file_slice(self):
        wav_data = None

        while wav_data is None or len(wav_data) != self.seq_len:
            try:
                file, file_length_sec, start_time, wav_data = self.try_random_file_slice()
            except Exception as e:
                logger.exception('Exception %s in random_file_slice.', e)

        # logger.debug('Sample: File: %s, File length: %s, Start time: %s',
        #              file, file_length_sec, start_time)

        return wav_data

    def try_random_file_slice(self):
        file = self._random_file()
        file_length_sec = self._file_length(file)
        segment_length_sec = self.seq_len / self.WAV_FREQ
        if file_length_sec < segment_length_sec:
            logger.warn('File "%s" has length %s, segment length is %s',
                        file, file_length_sec, segment_length_sec)

        start_time = random.random() * (file_length_sec - segment_length_sec * 2)  # just in case
        try:
            wav_data = self._file_slice(file, start_time)
        except Exception as e:
            logger.info(f'Exception in file slice: {e}. '
                        f'File: {file}, '
                        f'File length: {file_length_sec}, '
                        f'Start time: {start_time}')
            raise

        if len(wav_data) != self.seq_len:
            logger.warn('File "%s" has length %s, segment length is %s, wav data length: %s',
                        file, file_length_sec, segment_length_sec, len(wav_data))

        return file, file_length_sec, start_time, wav_data

    def dump_to_folder(self, output: Path, norm_db=False):
        for file_path in tqdm.tqdm(self.file_paths):
            output_file_path = output / file_path.relative_to(self.path).with_suffix('.h5')
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(suffix='.wav') as output_wav_file, \
                    NamedTemporaryFile(suffix='.wav') as norm_file_path, \
                    NamedTemporaryFile(suffix='.wav') as wav_convert_file:
                if norm_db:
                    logger.debug(f'Converting {file_path} to {wav_convert_file.name}')
                    subprocess.run(['ffmpeg',
                                    '-y',
                                    '-i', file_path,
                                    wav_convert_file.name],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

                    logger.debug(f'Companding {wav_convert_file.name} to {norm_file_path.name}')
                    subprocess.run(['sox',
                                    '-G',
                                    wav_convert_file.name,
                                    norm_file_path.name,
                                    'compand',
                                    '0.3,1',
                                    '6:-70,-60,-20',
                                    '-5',
                                    '-90',
                                    '0.2'],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
                    input_file_path = norm_file_path.name
                else:
                    input_file_path = file_path

                logger.debug(f'Converting {input_file_path} to {output_wav_file.name}')
                subprocess.run(['ffmpeg',
                                '-v', 'quiet',
                                '-y',  # overwrite
                                '-i', input_file_path,
                                # '-af', 'dynaudnorm',
                                '-f', 'wav',
                                '-ar', str(self.WAV_FREQ),  # audio rate
                                '-ac', '1',  # audio channels,
                                output_wav_file.name
                                ],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
                try:
                    rate, wav_data = wavfile.read(output_wav_file.name)
                except ValueError:
                    logger.info(f'Cannot read {file_path} wav conversion')
                    raise
                    # raise
                assert wav_data.dtype == np.int16
                wav = wav_data.astype('float')

                with h5py.File(output_file_path, 'w') as output_file:
                    chunk_shape = (min(10000, len(wav)),)
                    wav_dset = output_file.create_dataset('wav', wav.shape, dtype=wav.dtype,
                                                          chunks=chunk_shape)
                    wav_dset[...] = wav

                logger.debug(f'Saved input {file_path} to {output_file_path}. '
                             f'Wav length: {wav.shape}')


class H5Dataset(data.Dataset):
    def __init__(self, args, top, seq_len, dataset_name, epoch_len=10000, augmentation=None, short=False,
                 whole_samples=False, cache=False):
        #TODO send as argument
        self.args = args
        self.path = Path(top)
        self.seq_len = seq_len
        self.epoch_len = epoch_len
        self.short = short
        self.whole_samples = whole_samples
        self.augmentation = augmentation
        self.dataset_name = dataset_name

        self.file_paths = list(self.path.glob('**/*.h5'))
        if self.short:
            self.file_paths = [self.file_paths[0]]

        self.data_cache = {}
        if cache:
            for file_path in tqdm.tqdm(self.file_paths,
                                       desc=f'Reading dataset {top.parent.name}/{top.name}'):
                dataset = self.read_h5_file(file_path)
                self.data_cache[file_path] = dataset[:]

        if not self.file_paths:
            logger.error(f'No files found in {self.path}')

        logger.info(f'Dataset created. {len(self.file_paths)} files, '
                    f'augmentation: {self.augmentation is not None}. '
                    f'Path: {self.path}')

    def __getitem__(self, _):
        # pdb.set_trace()
        ret = None
        while ret is None:
            try:
                ret, midi = self.try_random_slice()
                if self.augmentation:
                    ret = [ret, self.augmentation(ret)]
                else:
                    ret = [ret, ret]

                if self.dataset_name == 'wav':
                    ret = [mu_law(x / 2 ** 15) for x in ret]
            except Exception as e:
                logger.info('Exception %s in dataset __getitem__, path %s', e, self.path)
                logger.debug('Exception in H5Dataset', exc_info=True)

        # print("getitem shapes: ", ret[0].shape, ret[1].shape, midi.shape)
        # if(midi is None):
        #     return torch.tensor(ret[0]), torch.tensor(ret[1]), None
        # return torch.tensor(ret[0]), torch.tensor(ret[1]), torch.LongTensor(midi)
        return ret[0], ret[1], midi

    def try_random_slice(self):
        h5file_path = random.choice(self.file_paths)
        # print("TESTING:",h5file_path)
        if h5file_path in self.data_cache:
            dataset = self.data_cache[h5file_path]
        else:
            dataset = self.read_h5_file(h5file_path)
        return self.read_wav_midi_data(dataset, h5file_path)

    def read_h5_file(self, h5file_path):
        try:
            f = h5py.File(h5file_path, 'r')
        except Exception as e:
            logger.exception('Failed opening %s', h5file_path)
            raise

        try:
            dataset = f[self.dataset_name]
        except Exception:
            logger.exception(f'No dataset named {self.dataset_name} in {file_path}. '
                             f'Available datasets are: {list(f.keys())}.')

        return dataset

    def read_midi_data(self, h5path, target_start_time, slice_len_in_sec):
        pkl_path = h5path.with_suffix(".pkl")
        if not os.path.exists(pkl_path):
            return None, None

        sTimes,notes,durations = pickle.load(open(pkl_path, 'rb'))

        target_end_time = target_start_time + slice_len_in_sec
        idx = 0

        if self.args.input_method=="notes_transition_only" or self.args.input_method=="notes_with_durations":
            # if self.args.input_type=="chords":
            #     target_chords, target_durations = np.zeros(16), np.zeros(16)
            # else:
            target_notes, target_durations = np.zeros((16, len(notes[0]))), np.zeros(16)
            for i,t in enumerate(sTimes):
                if t >= target_start_time:
                    if t <= target_end_time:
                        target_notes[idx] = notes[i]
                        target_durations[idx] = durations[i]
                        idx += 1

                if t > target_end_time or idx>=self.args.mode1_maxsize:
                    break
        
        if self.args.input_method=="notes_repeated_for_duration":
            time_to_idx = 100
            # if self.args.input_type == 'chords':
            #     target_chords, target_durations = np.zeros(int(slice_len*100)),np.zeros(int(slice_len*100))
            # else:
            target_notes = np.zeros((int(slice_len_in_sec * time_to_idx), len(notes[0])))
            target_durations = None #np.zeros((int(slice_len_in_sec * time_to_idx), len(notes[0])))
            # sTimes = sTimes - sTimes[0]
            # time_prev_in_ms = 0
            # for i,time in enumerate(sTimes[1:]):
            #     time_in_ms = int(time*100)
            #     target_chords[time_prev_in_ms:time_in_ms] = chords[i-1]
            #     time_prev_in_ms = time_in_ms
            # target_chords[time_in_ms:] = chords[-1]
            
            ###
            for i in range(len(sTimes)):
                if sTimes[i] + durations[i] > target_start_time and sTimes[i] < target_end_time:
                    start_idx = int(max(0, sTimes[i] - target_start_time) * time_to_idx)
                    end_idx = int(min(slice_len_in_sec, sTimes[i] + durations[i] - target_start_time) * time_to_idx)
                    if end_idx > start_idx:
                        target_notes[start_idx:end_idx] = [np.add(target_notes[j],notes[i]) for j in range(start_idx, end_idx)]
                if sTimes[i] > target_end_time:
                    break
            
            #TODO: Get SOS and EOS from vocab
            SOS = np.zeros((1,len(notes[0])))
            SOS[0][0] = 1
            EOS = np.zeros((1,len(notes[0])))
            EOS[0][1] = 1

            target_notes = np.concatenate((SOS, target_notes, EOS), axis=0)
            ###

        # print("target data: ", target_chords.shape, target_durations.shape)
        # target_notes: seq_len x vocab_size
        return target_notes, target_durations
    
    def read_wav_midi_data(self, dataset, path):
        if self.whole_samples:
            data = dataset[:]
        else:
            length = dataset.shape[0]

            if length <= self.seq_len:
                logger.debug('Length of %s is %s', path, length)
                
            # start_time = random.randint(0, length - self.seq_len)
            start_time_in_sec = random.randint(0, int((length - self.seq_len)/EncodedFilesDataset.WAV_FREQ))
            start_time = start_time_in_sec * EncodedFilesDataset.WAV_FREQ
            wav = dataset[start_time: start_time + self.seq_len]
            midi_chords, midi_durations = self.read_midi_data(path, start_time_in_sec, self.seq_len/EncodedFilesDataset.WAV_FREQ)
            assert wav.shape[0] == self.seq_len
        if(midi_chords is None):
            return wav.T, None
        # if self.mode==3:
        #     return wav.T, np.array(midi_chords) #seq_len x emb_size
        # else:
        return wav.T, np.array(midi_chords) #seq_len  x vocab_size


    # def read_wav_data(self, dataset, path):
    # we're not really using this function -> using the combined read_wav_midi_data instead
    #     if self.whole_samples:
    #         data = dataset[:]
    #     else:
    #         length = dataset.shape[0]

    #         if length <= self.seq_len:
    #             logger.debug('Length of %s is %s', path, length)

    #         start_time = random.randint(0, length - self.seq_len)
    #         data = dataset[start_time: start_time + self.seq_len]
    #         assert data.shape[0] == self.seq_len

    #     return data.T

    def __len__(self):
        return self.epoch_len


class WavFrequencyAugmentation:
    def __init__(self, wav_freq, magnitude=0.5):
        self.magnitude = magnitude
        self.wav_freq = wav_freq

    def __call__(self, wav):
        length = wav.shape[0]
        perturb_length = random.randint(length // 4, length // 2)
        perturb_start = random.randint(0, length // 2)
        perturb_end = perturb_start + perturb_length
        pitch_perturb = (np.random.rand() - 0.5) * 2 * self.magnitude

        ret = np.concatenate([wav[:perturb_start],
                              librosa.effects.pitch_shift(wav[perturb_start:perturb_end],
                                                          self.wav_freq, pitch_perturb),
                              wav[perturb_end:]])

        return ret


class DatasetSet:

    @staticmethod
    def my_collate(batch):
        x = torch.tensor([item[0] for item in batch])
        x_aug = torch.tensor([item[1] for item in batch])
        if batch[0][2] is None:
            return x, x_aug, None
        midi = torch.tensor([item[2] for item in batch])
        ## swap axis to get seq_len x batch_size x vocab_size
        midi = midi.permute(1,0,2).contiguous()
        # return [torch.tensor(x), torch.tensor(x_aug), torch.LongTensor(midi)]
        return x, x_aug, midi

    def __init__(self, dir: Path, seq_len, args):
        if args.data_aug:
            augmentation = WavFrequencyAugmentation(EncodedFilesDataset.WAV_FREQ, args.magnitude)
        else:
            augmentation = None

        self.train_dataset = H5Dataset(args, dir / 'train', seq_len, epoch_len=10000000000,
                                       dataset_name=args.h5_dataset_name, augmentation=augmentation,
                                       short=args.short, cache=False)
        
        self.train_loader = data.DataLoader(self.train_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            collate_fn=DatasetSet.my_collate,
                                            pin_memory=True)

        self.train_iter = iter(self.train_loader)

        self.valid_dataset = H5Dataset(args, dir / 'val', seq_len, epoch_len=1000000000,
                                       dataset_name=args.h5_dataset_name, augmentation=augmentation,
                                       short=args.short)
        self.valid_loader = data.DataLoader(self.valid_dataset,
                                            batch_size=args.batch_size,
                                            collate_fn=DatasetSet.my_collate,
                                            num_workers=args.num_workers // 10 + 1,
                                            pin_memory=True)

        self.valid_iter = iter(self.valid_loader)
