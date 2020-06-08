import random
def majority_vote(votes):
	'''
	xxx
	'''
	vote_counts = {}
	for vote in votes:
		if vote in vote_counts:
			vote_counts[vote] += 1
		else:
			vote_counts[vote] = 1
	winners = []
	max_count = max(vote_counts.values())
	for vote, count in vote_counts.items():
		if count == max_count:
			winners.append(vote)
	return random.choice(winners)



import scipy.stats as ss
def majority_vote_short(votes):
	'''Returns the most common element in votes'''
	mode, count = ss.mstats.mode(votes)
	return mode

def distance(p1,p2):
	return np.sqrt(np.sum(np.power(p2- p1,2)))

# ===== Starting with KNN =======
import numpy as np
points = np.array([[1,1], [1,2], [1,3], [2,1], [2,2], [2,3], [3,1], [3,2], [3,3]])


p = np.array([2.5, 2])
distances = np.zeros(points.shape[0]) # number of rows

for i in range(len(distances)):
	distances[i] = distance(p, points[i])

ind = np.argsort(distances)
#print(ind)

# algorithm
def find_nearest_neighbours(p, points, k= 5):
	'''
	Find the k nearest neighbours of point p and return their indices
	'''
	distances = np.zeros(points.shape[0])
	for i in range(len(distances)):
		distances[i] = distance(p, points[i])
	ind = np.argsort(distances)
	return ind[:k]

ind = find_nearest_neighbours(p, points, 3)
#print(points[ind])

# Visualizing the data
import matplotlib.pyplot as plt 
#plt.plot(points[:,0], points[:,1], 'ro')
#plt.plot(p[0], p[1], 'bo')
#plt.axis([0.5,3.5,0.5,3.5])
#plt.show()



def knn_predict(p ,points ,outcomes ,k= 5):
	ind = find_nearest_neighbours(p, points, k)
	return majority_vote(outcomes[ind])

outcomes = np.array([0,0,0,0,1,1,1,1,1])

#print(knn_predict(np.array([2.5,2.7]), points, outcomes, k=2))

#print(knn_predict(np.array([1,1]), points, outcomes, k=2))
