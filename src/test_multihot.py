# pdb.set_trace()
import numpy as np
import pickle
import os


pkl_path = "/datasets/tmp/dl4s/datasets/processed/maestro/preprocessed_notes/Johann_Sebastian_Bach/train/MIDI-Unprocessed_067_PIANO067_MID--AUDIO-split_07-07-17_Piano-e_3-03_wav--1.pkl"
# pkl_path = h5path.with_suffix(".pkl")
# if not os.path.exists(pkl_path):
#     return None, None

target_start_time = 15.3
slice_len_in_sec = 0.05
time_to_idx = 100
sTimes,notes,durations = pickle.load(open(pkl_path, 'rb'))
# print("time :", type(sTimes))
# print("durations: ", type(durations))

target_end_time = target_start_time + slice_len_in_sec
idx = 0
print("start time: ",target_start_time, " end time: ",target_end_time, "\n ")
target_notes = np.zeros((int(slice_len_in_sec*100),len(notes[0])))
# target_durations = np.zeros((int(slice_len*100),len(chords[0])))
print("len of chords: ", len(notes[0]))

# for i,t in enumerate(sTimes):
    # if t+durations[i] >= start_time and t < end_time:
    #     s = int(max(t-start_time, 0)*100)
    #     e = int(min(t + durations[i] - start_time, slice_len)*100)
    #     if e>s:
    #         print("t:",t)
    #         print("start index: ", s, "   end index: ", e)
    #         # print("adding this\n", [np.add(target_chords[j],chords[i]) for j in range(s,e)])
    #         target_chords[s:e] = [np.add(target_chords[j],chords[i]) for j in range(s,e)]
    #         print("selected chord: sTime={} | duration={} | chord={}".format(sTimes[i], durations[i], chords[i]))
    # if t>end_time:
    #     break

for i in range(len(sTimes)):
    if sTimes[i] + durations[i] > target_start_time and sTimes[i] < target_end_time:
        start_idx = int(max(0, sTimes[i] - target_start_time) * time_to_idx)
        end_idx = int(min(slice_len_in_sec, sTimes[i] + durations[i] - target_start_time) * time_to_idx)
        target_notes[start_idx:end_idx] = [np.add(target_notes[j],notes[i]) for j in range(start_idx, end_idx)]
    if sTimes[i] > target_end_time:
        break

SOS = np.zeros((1,len(notes[0])))
SOS[0][0] = 1
EOS = np.zeros((1,len(notes[0])))
EOS[0][1] = 1
print(SOS.shape, target_notes.shape, EOS.shape)

target_notes = np.concatenate((SOS, target_notes, EOS), axis=0)


print("Actual: ")
for t,n,d in zip(sTimes, notes, durations):
    if t+d > target_start_time and t < target_end_time:
        print(t,n,d)

target_notes.clip(0,1)
print("\n\nresult: \n", target_notes)
