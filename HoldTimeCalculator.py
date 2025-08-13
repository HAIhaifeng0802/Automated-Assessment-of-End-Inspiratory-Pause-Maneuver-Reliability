import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np
import math

# Load data
data = np.load(
    r"./filepath/filename.npy",
    allow_pickle=True,
).item()

waveData = np.array([data["p"], data["f"], data["v"]]).T
refSampleRate = data["refSampleRate"]
vwaveData = waveData[:, 2]

# Find breath-hold plateau 
def find_plateau(vwave_data, window_size=5):
    v_plat_begin = None
    v_plat_end = None
    
    # Find plateau start
    for m in range(window_size, len(vwave_data) - window_size):
        prev_window = vwave_data[m-window_size:m]
        next_window = vwave_data[m:m+window_size]
        
        if (prev_window[-1] - prev_window[0]) > (0.01*prev_window[-1]) and \
           abs(next_window[-1] - next_window[0]) < (0.01*next_window[0]):
            v_plat_begin = m
            break
    
    # Find plateau end        
    for m in range(window_size, len(vwave_data) - window_size):
        prev_window = vwave_data[len(vwave_data)-m-window_size : len(vwave_data)-m]
        next_window = vwave_data[len(vwave_data)-m : len(vwave_data)-m+window_size]
        
        if (next_window[-1] - next_window[0]) < -15 and \
           abs(prev_window[-1] - prev_window[0]) < 2:
            v_plat_end = len(vwave_data) - m
            break
    
    return v_plat_begin, v_plat_end

v_plat_begin, v_plat_end = find_plateau(vwaveData[10:-10])  # Remove edge artifacts
v_plat_begin += 10  # Adjust for removed edges
v_plat_end += 10

# Calculate hold time in seconds
if v_plat_begin and v_plat_end:
    hold_time = (v_plat_end - v_plat_begin + 1) / refSampleRate
    print(f"Breath-hold plateau detected from sample {v_plat_begin} to {v_plat_end}")
    print(f"Hold time: {hold_time:.2f} seconds")
else:
    print("No valid breath-hold plateau detected")

# Simplified visualization (optional)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(vwaveData, label='Volume')
if v_plat_begin and v_plat_end:
    plt.axvspan(v_plat_begin, v_plat_end, color='red', alpha=0.3, label='Hold Plateau')
plt.xlabel('Samples')
plt.ylabel('Volume (mL)')
plt.legend()
plt.show()