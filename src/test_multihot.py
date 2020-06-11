# pdb.set_trace()
import numpy as np
import pickle
import os


pkl_path = "/datasets/tmp/dl4s/datasets/processed/midi_wav/Johann_Sebastian_Bach/train/MIDI-Unprocessed_067_PIANO067_MID--AUDIO-split_07-07-17_Piano-e_3-03_wav--1.pkl"
# pkl_path = h5path.with_suffix(".pkl")
# if not os.path.exists(pkl_path):
#     return None, None

start_time = 9.95
slice_len = 0.06
sTimes,chords,durations = pickle.load(open(pkl_path, 'rb'))
# print("time :", type(sTimes))
# print("durations: ", type(durations))

end_time = start_time + slice_len
idx = 0
print("start time: ",start_time, " end time: ",end_time, "\n ")
target_chords = np.zeros((int(slice_len*100),len(chords[0])))
# target_durations = np.zeros((int(slice_len*100),len(chords[0])))
print("len of chords: ", len(chords[0]))

for i,t in enumerate(sTimes):
    if t+durations[i] >= start_time and t < end_time:
        s = int(max(t-start_time, 0)*100)
        e = int(min(t + durations[i] - start_time, slice_len)*100)
        if e>s:
            print("t:",t)
            print("start index: ", s, "   end index: ", e)
            # print("adding this\n", [np.add(target_chords[j],chords[i]) for j in range(s,e)])
            target_chords[s:e] = [np.add(target_chords[j],chords[i]) for j in range(s,e)]
            print("selected chord: sTime={} | duration={} | chord={}".format(sTimes[i], durations[i], chords[i]))
    if t>end_time:
        break
    
target_chords.clip(0,1)
print("\n\nresult: \n", target_chords)
