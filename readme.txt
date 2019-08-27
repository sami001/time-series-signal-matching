1. Unzip the file "time-series-signal-matching.zip"

2. On terminal/command prompt, cd to the location where you unzipped the file. Now run the following commands to reproduce my experiment results:

cd time-series-signal-matching-master
python trip_match.py mobile_trips.json obd2_trips.json 10 50 0.5 0.3


3. To test with any data set and any parameter values, the general command format is:

python trip_match.py <input_file1> <input_file2> <window_size> <interpolation_size> <slicing_factor> <match_threshold>

4. All figures of matching trip pairs will be generated in the "plots" folder
