import numpy as np 
import pandas as pd 
import scipy.stats as ss
import matplotlib.pyplot as plt 


n = 100

# Generating points
beta_0 = 5
beta_1 = 2
np.random.seed(1)

# [0-10]
x = 10 * ss.uniform.rvs(size= n)
y = beta_0 + beta_1 * x + ss.norm.rvs(loc= 0, scale= 1, size= n) # ss.norm is noise


rss = []
slopes = np.arange(-10,15,0.001)
for slope in slopes:
	rss.append(np.sum((y - beta_0 - slope *x)**2))

ind_min = np.argmin(rss)


print(ind_min, "ESTIMATE: ", slopes[ind_min])

#Plotting a figure
plt.figure()
plt.plot(slopes, rss)
plt.xlabel("Slope")
plt.ylabel("RSS")
plt.show()