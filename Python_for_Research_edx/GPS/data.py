import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

birddata = pd.read_csv('bird_tracking.csv', index_col = 0)

import datetime
#t1 = datetime.datetime.today()
#t2 = datetime.datetime.today()
#print(t2 - t1)

date_str = birddata.date_time[0]
#print(date_str[:-3])

# Formatting datetime

dt = datetime.datetime.strptime(date_str[:-3], "%Y-%m-%d %H:%M:%S")
#print(dt)

timestamps = []
for k in range(len(birddata)):
	timestamps.append(datetime.datetime.strptime\
		(birddata.date_time.iloc[k][:-3], "%Y-%m-%d %H:%M:%S"))

#print(timestamps[0:3])

birddata["timestamp"] = pd.Series(timestamps, index = birddata.index)
#print(birddata.head())
#
#
# Play with time

times = birddata.timestamp[birddata.bird_name == 'Eric']
elapsed_time = [time - times[0] for time in times]

print("Days passed: ",elapsed_time[1000] / datetime.timedelta(days = 1))

print("Hours passed: ",elapsed_time[1000] / datetime.timedelta(hours = 1))


plt.plot(np.array(elapsed_time) / datetime.timedelta(days=1))
plt.xlabel("Observation")
plt.ylabel("Elapsed time (days)")
plt.show()