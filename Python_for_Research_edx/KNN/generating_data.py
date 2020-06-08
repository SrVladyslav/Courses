import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

from fun import *

# Generating data
import scipy.stats as ss


def generate_synth_data(n= 50):
	'''
		Create two sets of points from bivariate normal distribution
	'''
	c1 = ss.norm(0,1).rvs((n,2))
	c2 = ss.norm(1,1).rvs((n,2))
	c = np.concatenate((c1, c2), axis= 0)

	outcomes = np.concatenate((np.repeat(0, n), np.repeat(1, n)))
	return (c, outcomes)

n = 20
(points, outcomes) = generate_synth_data(n)


# Plotting this points
#plt.figure()
#plt.plot(points[:n, 0], points[:n, 1], 'ro')
#plt.plot(points[n:, 0], points[n:, 1], 'bo')
#plt.show()


# Prediction grid

def make_prediction_grid(predictors, outcomes, limits, h, k):
	'''
	Classify each point on the prediction grid
	'''
	(x_min, x_max, y_min, y_max) = limits
	xs = np.arange(x_min, x_max, h)
	ys = np.arange(y_min, y_max, h)
	xx, yy = np.meshgrid(xs, ys)

	prediction_grid = np.zeros(xx.shape, dtype= int)
	for i, x, in enumerate(xs):
		for j, y in enumerate(ys):
			p = np.array([x,y])
			prediction_grid[j, i] = knn_predict(p, predictors, outcomes, k)
	return (xx, yy, prediction_grid)

# Plotting the prediction grid

def plot_prediction_grid (xx, yy, prediction_grid, filename= ''):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.title('Prueba K-NN 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.show()
    #plt.savefig(filename)
 
(predictors, outcomes) = generate_synth_data()

k = 2; limits = (-3,4,-3,4); h= 0.1
#(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits,h, k)
#plot_prediction_grid(xx, yy, prediction_grid, 'xD')


# SKELEARN====================================0
from sklearn import datasets

iris = datasets.load_iris()

predictors = iris.data[:, 0:2]
outcomes = iris.target

plt.plot(predictors[outcomes == 0][:,0],predictors[outcomes == 0][:,1], 'ro')
plt.plot(predictors[outcomes == 1][:,0],predictors[outcomes == 0][:,1], 'go')
plt.plot(predictors[outcomes == 2][:,0],predictors[outcomes == 0][:,1], 'bo')
#plt.show()

k = 5; limits = (4 ,8, 1.5, 4.5); h= 0.1
#(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits,h, k)
#plot_prediction_grid(xx, yy, prediction_grid, 'xD')

## KNN SciLearn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(predictors, outcomes)
sk_predictions = knn.predict(predictors)

#print(sk_predictions[0:10])

my_predictions = np.array([knn_predict(p, predictors, outcomes, 5) for p in predictors])

# How often the SciKit predictions are the same that my homemade algorithm?

same = np.mean(sk_predictions == my_predictions) * 100
print(same, "% are the same")
print("SkLearn accuracy: ", round(np.mean(sk_predictions == outcomes) * 100,3), "%")
print("My homemade accuracy: ",round(np.mean(my_predictions == outcomes) * 100,3), "%")