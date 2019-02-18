import sys
import time
import json
import math
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

    '''
    this method calculates distance between two time serieses using dynamic time warping
    returns minimum distance
    '''
    def dtw(self, time_series1, time_series2):
        DTW = {}

        len1 = time_series1.size
        len2 = time_series2.size
        '''
        to handle the situation if the predefined window size is larger than the difference of
        two timeseries lengths, the searched timestamp index of the second time series may go out of range.
        '''
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
        return math.sqrt(DTW[(len1-1, len2-1)])

    '''
    this method handles outliers by replacing the speed data points with the help of the accuracy values
    returns speed data with minimized effect of outliers
    '''
    def outlier_handler(self, speeds_mobile, accuracy):
        print('Working on data outliers ...')
        speed_mobile_clean = deepcopy(speeds_mobile)
        for i in range(len(speed_mobile_clean)):
            #normalizing the accuracy data of a single trip
            min_accuracy = min(accuracy[i])
            max_accuracy = max(accuracy[i])
            #increasing max_accuracy value when |max_accuracy-min_accuracy| is low
            #to avoid icorrectly classifying non-outliers as outliers
            if((max_accuracy - min_accuracy) < 50):
                max_accuracy += 100
            accuracy[i] = [((n - min_accuracy) / (max_accuracy - min_accuracy)) for n in accuracy[i]]
            '''
            Adjusting the amplitude of the graph modifying the speed data points based on the reading accuracy.
            An estimate for the original speed can be either the mean of the speed values (for the first data point)
            or the value of previous speed data point. The difference between the data speed and estimated speed
            is then multiplied by the accruacy data to get the adjustment value. This value is then added to the
            speed data to get the final adjusted speed
            '''
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

    '''
    this method interpolates the time versus speed data for a reduced timeseries
    returns interpolated reduced size time and speed data
    '''
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

    '''
    this method loads data and does preprocessing
    returns processed data
    '''
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
            '''
            gets reduced trip data by interpolating speed for fewer timestamps than the original data
            for efficient computation, while the timestamp interval remains unchanged
            '''
            if('accuracy' in data[0][0]):
                #for device having "accuracy" data, extract the accuracy value
                accuracy = [[dict['accuracy'] for dict in d] for d in data]
                #reducing the effect of outliers using device accuracy readings
                speeds_minimized_outliers = self.outlier_handler(speeds, accuracy)
                timestamps_intrpld, speeds_intrpld = self.interpolation(timestamps, speeds_minimized_outliers , self.interpolation_size)
            else:
                #for device having no "accuracy" data
                timestamps_intrpld, speeds_intrpld = self.interpolation(timestamps, speeds, self.interpolation_size)
        return timestamps, speeds, timestamps_intrpld, speeds_intrpld

    '''
    this method calculates the mean of those speed values 
    which are larger than the mean speed of a trip data
    '''
    def filtered_mean(self, speeds):
        #calculating the mean of the entire speed data of a trip
        sum_value =  sum([sum(l) for l in speeds])
        count = sum([len(l) for l in speeds])
        first_mean = sum_value/count

        #get a new speed data filtering speed values larger than first_mean, and
        #then calculate the mean of the new speeds
        sum_value = sum([sum([le for le in l if le > first_mean]) for l in speeds])
        count = sum(map(sum, [[le > first_mean for le in l] for l in speeds]))
        return sum_value/count

    '''
    this method returns the speed scaling factor
    '''
    def get_speed_scaling_factor(self, speeds_mobile, speeds_obd2):
        return self.filtered_mean(speeds_obd2)/self.filtered_mean(speeds_mobile)

    '''
    this method matches a trip of dataset A with a trip of dataset B
    returns for every trip of A: its best match with a trip of B
    '''
    def match_time_series_pairs(self, times_A, speeds_A, times_B, speeds_B, text):
        print('Finding trip match pairs ......')
        match = []

        #iterating over all trips of dataset A
        for idx_A, speed_A in enumerate(speeds_A):
            res = 0 #best match
            start_time = 0 #index of the starting timestamp of trip B
            min_dtw = float("inf") #minimum dynamic time warping distance
            # iterating over all trips of dataset B
            for idx_B, speed_B in enumerate(speeds_B):
                '''
                to compute distance of trip A and trip B,
                slicing over the timestamps of trip B
                as "starting point" of its time series.

                slicing_factor (a value between 0 and 1.0) determines after what percentage of trip B data
                we will stop slicing. Larger value of slicing_factor generates small slices of trip B,
                thereby results in tiny matched intervals, which may not be desirable.
                '''

                for k in range(int(self.slicing_factor*speed_B.size)):
                    interval_match = min(times_B[idx_B][k:][-1], times_A[idx_A][-1]) - times_B[idx_B][k:][0]
                    interval_A     = times_A[idx_A][-1] - times_A[idx_A][0]
                    '''
                    match_threshold is the minimum percentage of overlap we are
                    allowing between a pair of trip to qualify it as a 'match'.
                    '''
                    if((interval_match > self.match_threshold * interval_A) and (interval_A > self.match_threshold * interval_match)):
                        cost = self.dtw(speed_A, speed_B[k:])
                        if (cost < min_dtw):
                            min_dtw = cost
                            res = idx_B
                            start_time = k
            '''
            storing the match result for a trip (index of the trip with best score, the similarity score, index of the
            #starting timestamp of the matched trip, a text showing the order of comparison)
            '''
            match.append([res, min_dtw, start_time, text])

            #when mobile data is A, OBDII data is B; so we are slicing OBDII trips
            if(text == 'slicing_obd2'):
                print('Mobile trip ' + str(idx_A) + ' found a match with an OBDII trip.')
            #when OBDII data is A, mobile data is B; so we are slicing mobile trips
            else:
                print('OBDII trip ' + str(idx_A) + ' found a match with a mobile trip.')
        return match

    '''
    this method plot a best match matched trip pair on the same graph
    '''
    def plot_match(self, x1, y1, x2, y2, match, idx):
        plt.title('Mobile trip #' + str(idx) + ' has a match with Obd2 trip #' + str(match[idx][0]))
        #drawing two vertical black lines to indicate the matched interval
        plt.axvline(x=(x1[0] if x1[0] > x2[0] else x2[0]), color='k', lw=0.5)
        plt.axvline(x=(x1[-1] if x1[-1] < x2[-1] else x2[-1]), color='k', lw=0.5)
        plt.plot(x1, y1, 'g', lw=1, label='mobile trip')
        plt.plot(x2, y2, 'r', lw=1, label='obd2 trip')
        plt.xlabel('time')
        plt.ylabel('speed')
        plt.legend()
        plt.savefig('plots/mobile_trip_' + str(idx) + '.png')
        plt.clf()

    '''
    this is the parent method which evokes all other methods
    '''
    def match_trips(self):
        print('Loading Mobile data ...')
        print('Interpolating Mobile data ...')
        #getting the clean data and clean interpolated data from the data dump
        timestamps_mobile, speeds_mobile, timestamps_mobile_intrpld, speeds_mobile_intrpld \
            = self.data_preprocess(self.filename1)
        
        print('Loading OBDII data ...')
        print('Interpolating OBDII data ...')
        timestamps_obd2, speeds_obd2, timestamps_obd2_intrpld, speeds_obd2_intrpld \
            = self.data_preprocess(self.filename2)

        print('Calculating scaling factor ...')
        #computing a scaling factor to convert mobile speed unit to the OBDII speed unit
        speed_scaling_factor = self.get_speed_scaling_factor(speeds_mobile, speeds_obd2)

        #scaling the mobile device speed for both the original data and interpolated data to match OBDII speeds
        speeds_mobile         = [[(le*speed_scaling_factor) for le in l] for l in speeds_mobile]
        speeds_mobile_intrpld = [l*speed_scaling_factor for l in speeds_mobile_intrpld]

        #for every mobile trip, finding its best match with a OBDII trip
        mobile_match = self.match_time_series_pairs(timestamps_mobile_intrpld, speeds_mobile_intrpld,
                                                    timestamps_obd2_intrpld, speeds_obd2_intrpld, 'slicing_obd2')
        # for every OBDII trip, finding its best match with a mobile trip
        obd2_match   = self.match_time_series_pairs(timestamps_obd2_intrpld, speeds_obd2_intrpld,
                                                    timestamps_mobile_intrpld, speeds_mobile_intrpld, 'slicing_mobile')
        
        print('Updating matched trip pair list ...')
        #updating the results in mobile_match if any better match was found in obd2_match
        for i in range(len(obd2_match)):
            if(mobile_match[obd2_match[i][0]][0] != i and mobile_match[obd2_match[i][0]][1] > obd2_match[i][1]):
                mobile_match[obd2_match[i][0]] = [i, obd2_match[i][1], obd2_match[i][2], 'slicing_mobile']
        
        print('Plotting graph and generating output figures ...')
        #plotting each mobile trip and its best matched OBDII trip
        for i in range(len(mobile_match)):
            t1 = timestamps_mobile[i]
            s1 = speeds_mobile[i]
            t2 = timestamps_obd2[mobile_match[i][0]]
            s2 = speeds_obd2[mobile_match[i][0]]

            #shifting the plot along the time axis, so both plots are FULLY visible on the same graph
            if(mobile_match[i][3] == 'slicing_obd2'):
                t1 = [(l + timestamps_obd2_intrpld[mobile_match[i][0]][mobile_match[i][2]]) for l in t1]
            else:
                t2 = [(l + timestamps_mobile_intrpld[mobile_match[i][0]][mobile_match[i][2]]) for l in t2]

            #plot the mathing pair
            self.plot_match(t1, s1, t2, s2, mobile_match, i)

        print("Completed in %s seconds !!!" % (time.time() - self.start_time))



if __name__ == "__main__":
    assert isinstance(sys.argv[1], str)
    assert isinstance(sys.argv[2], str)
    assert isinstance(sys.argv[3], str)
    assert isinstance(sys.argv[4], str)
    assert isinstance(sys.argv[5], str)

    obj = TripMatch(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    obj.match_trips()
