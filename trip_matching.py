#get rid of outliers from ACCURACY

import pandas as pd
import time
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import statistics
from copy import deepcopy
from scipy.interpolate import UnivariateSpline, pchip



start_time = time.time()

timestamps_mobile = []
timestamps_obd2 = []
speeds_mobile = []
speeds_obd2 = []


timestamps_mobile_splined = []
timestamps_obd2_splined = []
speeds_mobile_splined = []

speeds_obd2_splined = []
accuracy_mobile = []

avg_speed_mobile = 0
avg_speed_obd2 = 0

def dtw(time_series1, time_series2):
    DTW = {}

    len1 = time_series1.size
    len2 = time_series2.size

    w = max(10, abs(len1 - len2))

    for i in range(-1, len1):
        for j in range(-1, len2):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len1):
        for j in range(max(0, i - w), min(len2, i + w)):
            dist = (time_series1[i] - time_series2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
    return DTW[(len1-1, len2-1)]

with open('C:/root_telematics_data/mobile_trips.json') as json_file:
    data = json.load(json_file)
    sum_value = 0
    count = 0
    min_accuracy = float("inf")
    max_accuracy = float("-inf")

    for i in range(len(data)):
        t=[]
        s=[]
        a=[]
        for j in range(len(data[i])):
            if(j == 0 or data[i][j]['timestamp'] > t[-1]):
                t.append(data[i][j]['timestamp'])
                s.append(data[i][j]['speed'])
                a.append(data[i][j]['accuracy'])
                sum_value += s[-1]
                count += 1
        t[:] = [x - min(t) for x in t]
        timestamps_mobile.append(t)
        speeds_mobile.append(s)
        a = [(n - min(a))/(max(a) - min(a)) for n in a] #normalizing with row min and max
        accuracy_mobile.append(a)

    speed_mobile_no_outlier = deepcopy(speeds_mobile)
    print(speeds_mobile[0][0])
    speed_mobile_no_outlier[0][0] = 1
    print(speeds_mobile[0][0])


    for i in range(len(accuracy_mobile)):
        for j in range(len(accuracy_mobile[i])):
            if(j == 0):
                adj = (statistics.mean(speed_mobile_no_outlier[i]) - speed_mobile_no_outlier[i][j]) * accuracy_mobile[i][j];
            else:
                adj = (speed_mobile_no_outlier[i][j-1] - speed_mobile_no_outlier[i][j]) * accuracy_mobile[i][j]

            speed_mobile_no_outlier[i][j] = speed_mobile_no_outlier[i][j] + adj

        spl = pchip(timestamps_mobile[i], speed_mobile_no_outlier[i])
        ts = np.linspace(min(timestamps_mobile[i]), max(timestamps_mobile[i]), 50)
        ss = spl(ts)
        timestamps_mobile_splined.append(ts)
        speeds_mobile_splined.append(ss)

    avg = sum_value/count
    sum_value = 0
    count = 0

    for l in speeds_mobile:
        sum_value += sum(le for le in l if le > avg)
        count += sum(le > avg for le in l)
    avg_speed_mobile = sum_value/count


with open('C:/root_telematics_data/obd2_trips.json') as json_file:
    data = json.load(json_file)
    sum_value = 0
    count = 0
    for i in range(len(data)):
        t = []
        s = []
        for j in range(len(data[i])):
            if (j == 0 or data[i][j]['timestamp'] > t[-1]):
                t.append(data[i][j]['timestamp'])
                s.append(data[i][j]['speed'])
                sum_value += data[i][j]['speed']
                count += 1
        t[:] = [x - min(t) for x in t]
        timestamps_obd2.append(t)
        speeds_obd2.append(s)

        spl = pchip(t, s)
        ts = np.linspace(min(t), max(t), 50)
        ss = spl(ts)
        timestamps_obd2_splined.append(ts)
        speeds_obd2_splined.append(ss)

    avg = sum_value/count
    sum_value = 0
    count = 0

    for l in speeds_obd2:
        sum_value += sum(le for le in l if le > avg)
        count += sum(le > avg for le in l)
    avg_speed_obd2 = sum_value/count
    scaling_factor = avg_speed_obd2/avg_speed_mobile
#print(scaling_factor)

speeds_mobile_splined = [l*scaling_factor for l in speeds_mobile_splined] #scaling the speed values for the splined equation

mobile_match = []

for idx_mobile, speed_mobile in enumerate(speeds_mobile_splined):
    ans = 0
    start = 0
    min_dtw =  float("inf")
    for idx_obd2, speed_obd2 in enumerate(speeds_obd2_splined):
        for k in range(int(speed_obd2.size/2)):
            cost = dtw(speed_mobile, speed_obd2[k:])
            if (cost < min_dtw):
                min_dtw = cost
                ans = idx_obd2
                start = k

    mobile_match.append([ans, min_dtw, start, 'sliding_obd2'])
    print('found a match for mobile trip ' + str(idx_mobile))

obd2_match = []

for idx_obd2, speed_obd2 in enumerate(speeds_obd2_splined):
    ans = 0
    start = 0
    min_dtw =  float("inf")
    for idx_mobile, speed_mobile in enumerate(speeds_mobile_splined):
        for k in range(int(speed_mobile.size/2)):
            cost = dtw(speed_obd2, speed_mobile[k:])
            if (cost < min_dtw):
                min_dtw = cost
                ans = idx_mobile
                start = k

    obd2_match.append([ans, min_dtw, start])
    print('found a match for obd2 trip ' + str(idx_obd2))

#checking if there's any better match
for i in range(len(obd2_match)):
    if(mobile_match[obd2_match[i][0]][0] != i and mobile_match[obd2_match[i][0]][1] > obd2_match[i][1]):
        mobile_match[obd2_match[i][0]][0] = i
        mobile_match[obd2_match[i][0]][1] = obd2_match[i][1]
        mobile_match[obd2_match[i][0]][2] = obd2_match[i][2]
        mobile_match[obd2_match[i][0]][3] = 'sliding_mobile'

for i in range(len(mobile_match)):
    x1 = timestamps_mobile[i]
    y1 =  [(l*scaling_factor) for l in speeds_mobile[i]]
    x2 = timestamps_obd2[mobile_match[i][0]]
    y2 = speeds_obd2[mobile_match[i][0]]

    if(mobile_match[i][3] == 'sliding_obd2'):
        x1 = [(l + timestamps_obd2_splined[mobile_match[i][0]][mobile_match[i][2]]) for l in x1]
    else:
        x2 = [(l + timestamps_mobile_splined[mobile_match[i][0]][mobile_match[i][2]]) for l in x2]


    plt.plot(x1, y1, 'g', lw=1, label='mobile trip')
    plt.plot(x2, y2, 'r', lw=1, label='obd2 trip')
    plt.xlabel('time')
    plt.ylabel('speed')
    plt.legend()
    plt.savefig('plots/mobile_trip_'+str(i)+'.png')
    plt.clf()

print("--- %s seconds ---" % (time.time() - start_time))
