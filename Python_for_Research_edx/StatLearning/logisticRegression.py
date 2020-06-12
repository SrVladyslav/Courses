import numpy as np 
import scipy.stats as ss 
import matplotlib.pyplot as plt 
 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

h = 1
sd = 1 # standart derivation
n = 50 # observations

def gen_data(n, h, sd1, sd2):
	x1 = ss.norm.rvs(-h, sd1, n)
	y1 = ss.norm.rvs(0, sd1, n)

	x2 = ss.norm.rvs(h, sd2, n)
	y2 = ss.norm.rvs(0, sd2, n)
	return (x1,y1,x2,y2)

(x1,y1,x2,y2) = gen_data(50, 1, 1, 1.5)

(x1,y1,x2,y2) = gen_data(1000, 1.5, 1, 1.5)

def plot_data(x1,y1,x2,y2):
	plt.figure()
	plt.plot(x1,y1,'o',ms=2)
	plt.plot(x2,y2,'o',ms=2)
	plt.xlabel("$X_1$")
	plt.ylabel("$X_2$")
	plt.show()

#plot_data(x1,y1,x2,y2)


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression() #Model

# Vertical stack
X = np.vstack((np.vstack((x1,  y1)).T, np.vstack((x1,  y1)).T))

#outputs
n = 1000
y = np.hstack((np.repeat(1,n), np.repeat(2, n)))
#print(y.shape)

# data split
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size= 0.5, random_state= 1)
print(X_train.shape)

clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# probs by classes
print(clf.predict_proba(np.array([-2,0]).reshape(1,-1)))