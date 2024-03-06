####################################################################################################
# DTW算出
####################################################################################################


from myapp import app
import csv
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from matplotlib import pyplot as plt





# DTW距離
def linear_dtw(wave1, wave2):
    x = np.array(wave1).reshape(-1, 1)
    y = np.array(wave2).reshape(-1, 1)
    distance, _ = fastdtw(x, y, dist=euclidean)
    return distance

# 非線形伸縮後のDTW距離
def nonlinear_dtw(wave1, wave2):
    x = np.array(wave1).reshape(-1, 1)
    y = np.array(wave2).reshape(-1, 1)
    _, path = fastdtw(x, y, dist=euclidean)
    y_align, _ = nonlinear_alignment(x, y, path)
    distance, _ = fastdtw(x, y_align, dist=euclidean)
    return distance

# 非線形伸縮
def nonlinear_alignment(base_wave, origin_wave, path):
    align_wave = np.zeros(len(base_wave))
    for (i, k) in path:
        align_wave[i] = origin_wave[k]
    return align_wave, origin_wave

