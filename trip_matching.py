import sys
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import statistics
from copy import deepcopy
from scipy.interpolate import pchip

class TripMatch:
    #initializing the command-line arguments
    def __init__(self, filename1, filename2, window_size, interpolation_size, slicing_factor, match_threshold):
        self.start_time = time.time()
        self.filename1 = filename1
        self.filename2 = filename2
        self.window_size = int(window_size)
        self.interpolation_size = int(interpolation_size)
        self.slicing_factor = float(slicing_factor)
        self.match_threshold = float(match_threshold)

    #calculating distance between two time serieses using dynamic time warping
    #returns minimum distance
    def dtw(self, time_series1, time_series2):
        DTW = {}

        len1 = time_series1.size
        len2 = time_series2.size

        #to handle the situation if the predefined window size is larger than the difference of
        #two timeseries lengths, the searched timestamp index of the second time series may go out of range.
        w = max(self.window_size, abs(len1 - len2))

        #initializing the dynamic programming table
        for i in range(-1, len1):
            for j in range(-1, len2):
                DTW[(i, j)] = float('inf')
        DTW[(-1, -1)] = 0

        for i in range(len1):
            #using a fixed window size of w to search for a match
            for j in range(max(0, i - w), min(len2, i + w)):
                dist = (time_series1[i] - time_series2[j]) ** 2
                DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
        return DTW[(len1-1, len2-1)]

    #handing outliers by replacing the speed data points with the help of the accuracy values
    #returns speed data with minimized effect of outliers
    def outlier_handler(self, speeds_mobile, accuracy):

        speed_mobile_clean = deepcopy(speeds_mobile)
        for i in range(len(speed_mobile_clean)):
            #normalizing the accuracy data of a single trip
            accuracy[i] = [(n - min(accuracy[i])) / (max(accuracy[i]) - min(accuracy[i])) for n in accuracy[i]]

            #adjusting the amplitude of the graph modifying the speed data points based on the reading accuracy
            #an estimate for the original speed can be either the mean of the speed values (for the first data point)
            #or the value of previous speed data point. The difference between the speed data and estimated speed
            #is then multiplied by the accruacy data to get the adjustment value. This value is then added to the
            #speed data to get the final adjusted speed data
            for j in range(len(speed_mobile_clean[i])):
                #finding the adjustment distance for the first speed data point
                if(j == 0):
                    adj = (statistics.mean(speed_mobile_clean[i]) - speed_mobile_clean[i][j]) * accuracy[i][j];
                #finding the adjustment distance for remaining data points comparing each with the previous data point
                else:
                    adj = (speed_mobile_clean[i][j-1] - speed_mobile_clean[i][j]) * accuracy[i][j]
                #adjusting speed data
                speed_mobile_clean[i][j] = speed_mobile_clean[i][j] + adj
        return speed_mobile_clean

    #interpolating the time versus speed data for a reduced timeseries
    #returns interpolated reduced size time and speed data
    def interpolation(self, times, speeds, num_points):
        times_intrpld =[]
        speeds_intrpld = []
        for i in range(len(speeds)):
            #piecewise cubic hermite interpolating polynomial for the time and speed data
            pch = pchip(times[i], speeds[i])
            #creating a reduced number of evenly spaced timestamps over the original interval
            ts = np.linspace(times[i][0], times[i][-1], num_points)
            #calculating speed data for the new timestamps using the polynomial
            ss = pch(ts)
            times_intrpld.append(ts)
            speeds_intrpld.append(ss)
        return times_intrpld, speeds_intrpld

    #loads data and does preprocessing
    #returns processed data
    def data_preprocess(self, filename):
        timestamps = []
        speeds = []
        #reads data from json file
        with open(filename) as json_file:
            data = json.load(json_file)
            sum_value = 0 #sum of the speed values of the entire data set
            count = 0     #total number of speed values in the entire data set
            for i in range(len(data)):
                t = []
                s = []
                for j in range(len(data[i])):
                    if (j == 0 or data[i][j]['timestamp'] > t[-1]):
                        t.append(data[i][j]['timestamp'])
                        s.append(data[i][j]['speed'])
                        sum_value += s[-1]
                        count += 1
                #shifts all timestamps to left so that the time series starts from zero
                t = [x - min(t) for x in t]
                timestamps.append(t)
                speeds.append(s)

            #gets reduced trip data by interpolating speed for fewer timestamps than the original data
            #for efficient computation, while the timestamp interval remains unchanged
            if('accuracy' in data[0][0]):
                #for device having "accuracy" data, extract the accuracy value
                accuracy = [[dict['accuracy'] for dict in d] for d in data]
                #reducing the effect of outliers using device accuracy readings
                speeds_minimized_outliers = self.outlier_handler(speeds, accuracy)
                timestamps_intrpld, speeds_intrpld = self.interpolation(timestamps, speeds_minimized_outliers , self.interpolation_size)
            else:
                #for device having no "accuracy" data
                timestamps_intrpld, speeds_intrpld = self.interpolation(timestamps, speeds, self.interpolation_size)

            #the mean speed of entire dataset is calculated as first_mean
            first_mean = sum_value / count
            #final_mean is the mean of speed values which are greater than first_mean
            sum_value = sum([sum([le for le in l if le > first_mean]) for l in speeds])
            count = sum(map(sum, [[le > first_mean for le in l] for l in speeds]))
            final_mean = sum_value / count

        return timestamps, speeds, timestamps_intrpld, speeds_intrpld, final_mean

    #matches a trip of dataset A with a trip of dataset B
    #returns for every trip of A, its best match with a trip of B
    def match_time_series_pairs(self, times_A, speeds_A, times_B, speeds_B, text):
        match = []

        #iterating over all trips of dataset A
        for idx_A, speed_A in enumerate(speeds_A):
            res = 0 #best match
            start_time = 0 #index of the starting timestamp of trip B
            min_dtw = float("inf") #minimum dynamic time warping distance
            # iterating over all trips of dataset B
            for idx_B, speed_B in enumerate(speeds_B):
                #to compute distance of trip A and trip B,
                #slicing over the timestamps of trip B
                #as "starting point" of its time series.

                #slicing_factor (a value between 0 and 1.0) determines after what percentage of trip B data
                #we will stop slicing. Larger value of slicing_factor generates small slices of trip B,
                #thereby results in tiny matched intervals, which may not be desirable.

                for k in range(int(self.slicing_factor*speed_B.size)):
                    interval_match = min(times_B[idx_B][k:][-1], times_A[idx_A][-1]) - times_B[idx_B][k:][0]
                    interval_A     = times_A[idx_A][-1] - times_A[idx_A][0]
                    #match_threshold represents the minimum percentage of overlap we are
                    #allowing between pairs to qualify it as a 'match'.
                    if((interval_match > self.match_threshold * interval_A) and (interval_A > self.match_threshold * interval_match)):
                        cost = self.dtw(speed_A, speed_B[k:])
                        if (cost < min_dtw):
                            min_dtw = cost
                            res = idx_B
                            start_time = k

            match.append([res, min_dtw, start_time, text])

            #when mobile data is A, OBDII data is B; so were are slicing OBDII trips
            if(text == 'slicing_obd2'):
                print('found a match for mobile trip ' + str(idx_A))
            #when OBDII data is A, mobile data is B; so we were slicing mobile trips
            else:
                print('found a match for obd2 trip ' + str(idx_A))
        return match

    #plot a best match matched trip pair on the same graph
    def plot_match(self, x1, y1, x2, y2, match, idx):
        plt.title('Mobile trip #' + str(idx) + ' has a match with Obd2 trip #' + str(match[idx][0]))
        plt.axvline(x=(x1[0] if x1[0] > x2[0] else x2[0]), color='k', lw=0.5)
        plt.axvline(x=(x1[-1] if x1[-1] < x2[-1] else x2[-1]), color='k', lw=0.5)
        plt.plot(x1, y1, 'g', lw=1, label='mobile trip')
        plt.plot(x2, y2, 'r', lw=1, label='obd2 trip')
        plt.xlabel('time')
        plt.ylabel('speed')
        plt.legend()
        plt.savefig('plots/mobile_trip_' + str(idx) + '.png')
        plt.clf()

    def match_trips(self):
        #getting the original data, interpolated data and mean speed from a data file
        timestamps_mobile, speeds_mobile, timestamps_mobile_intrpld, speeds_mobile_intrpld, avg_speed_mobile \
            = self.data_preprocess(self.filename1)
        timestamps_obd2, speeds_obd2, timestamps_obd2_intrpld, speeds_obd2_intrpld, avg_speed_obd2 \
            = self.data_preprocess(self.filename2)

        #computing a scaling factor to convert mobile speed unit to the OBDII speed unit
        speed_scaling_factor = avg_speed_obd2/avg_speed_mobile

        #scaling the mobile device speed for both the original data and interpolated data to match OBDII speeds
        speeds_mobile         = [[(le*speed_scaling_factor) for le in l] for l in speeds_mobile]
        speeds_mobile_intrpld = [l*speed_scaling_factor for l in speeds_mobile_intrpld]

        #for every mobile trip, finding its best match with a OBDII trip
        mobile_match = self.match_time_series_pairs(timestamps_mobile_intrpld, speeds_mobile_intrpld,
                                                    timestamps_obd2_intrpld, speeds_obd2_intrpld, 'slicing_obd2')
        # for every OBDII trip, finding its best match with a mobile trip
        obd2_match   = self.match_time_series_pairs(timestamps_obd2_intrpld, speeds_obd2_intrpld,
                                                    timestamps_mobile_intrpld, speeds_mobile_intrpld, 'slicing_mobile')

        #updating the results in mobile_match if any better match was found in obd2_match
        for i in range(len(obd2_match)):
            if(mobile_match[obd2_match[i][0]][0] != i and mobile_match[obd2_match[i][0]][1] > obd2_match[i][1]):
                mobile_match[obd2_match[i][0]] = [i, obd2_match[i][1], obd2_match[i][2], 'slicing_mobile']

        #plotting each mobile trip and its best matched OBDII trip
        for i in range(len(mobile_match)):
            t1 = timestamps_mobile[i]
            s1 = speeds_mobile[i]
            t2 = timestamps_obd2[mobile_match[i][0]]
            s2 = speeds_obd2[mobile_match[i][0]]

            #shifting the plot along the time axis, so both plots are visible on the graph with their original interval
            if(mobile_match[i][3] == 'slicing_obd2'):
                t1 = [(l + timestamps_obd2_intrpld[mobile_match[i][0]][mobile_match[i][2]]) for l in t1]
            else:
                t2 = [(l + timestamps_mobile_intrpld[mobile_match[i][0]][mobile_match[i][2]]) for l in t2]

            #plot the mathing pair
            self.plot_match(t1, s1, t2, s2, mobile_match, i)

        print("--- %s seconds ---" % (time.time() - self.start_time))



if __name__ == "__main__":
    assert isinstance(sys.argv[1], str)
    assert isinstance(sys.argv[2], str)

    obj = TripMatch(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    obj.match_trips()
