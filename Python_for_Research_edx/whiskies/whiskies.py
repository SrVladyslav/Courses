import pandas as pd
import numpy as np 

whisky = pd.read_csv("whiskies.txt")
whisky["Region"] = pd.read_csv("regions.txt")

# ROWS 0:10 and COLUMNS 0:5
#print(whisky.iloc[0:10, 0:5])

flavors = whisky.iloc[:, 2:14]
#print(flavors)

# Correlations
corr_flavors = pd.DataFrame.corr(flavors)
#print(corr_flavors)

import matplotlib.pyplot as plt 
'''
plt.figure(figsize = (10,10))
plt.pcolor(corr_flavors)
plt.colorbar()
plt.show()
'''
corr_whisky = pd.DataFrame.corr(flavors.transpose())

'''
print(corr_whisky)

plt.figure(figsize = (10,10))
plt.pcolor(corr_whisky)
plt.axis("tight")
plt.colorbar()
plt.show()
'''


# Spectral co clustering

from sklearn.cluster import SpectralCoclustering

model = SpectralCoclustering(n_clusters= 6, random_state= 0) 
model.fit(corr_whisky) # Data from the correlation matrix

# Every row corresponds to the cluster, every column 
# to the data parameter
print( np.sum(model.rows_, axis= 1) ) # Sumamos las columnas

# How many clusters belonging from each element
print( np.sum(model.rows_, axis= 0) )

# Each element from the array positions belongs from the number
# from this position
print(model.row_labels_)


# Comparing the correlation tables
whisky['Group'] = pd.Series(model.row_labels_, index = whisky.index)

# Reordering the group by 
whisky = whisky.iloc[np.argsort(model.row_labels_)] 

# Recalculate the correlation matrix again
whisky = whisky.reset_index(drop= True)

correlations = pd.DataFrame.corr(whisky.iloc[:, 2:14].transpose())

correlations = np.array(correlations)

# Plotting

plt.figure(figsize=(14,7))
plt.subplot(121)
plt.pcolor(corr_whisky)
plt.title("Original")
plt.axis("tight")
plt.subplot(122)
plt.pcolor(correlations)
plt.title("Rearranged")
plt.axis("tight")
plt.show()