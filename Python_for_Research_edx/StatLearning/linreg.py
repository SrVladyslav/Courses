import numpy as np 
import pandas as pd 
import scipy.stats as ss
import matplotlib.pyplot as plt 
import statsmodels.api as sm 


n = 500
beta_0 = 5
beta_1 = 2
beta_2 = -1

np.random.seed(1)
x_1 = 10 * ss.uniform.rvs(size= n)
x_2 = 10 * ss.uniform.rvs(size= n)

y = beta_0 + beta_1 * x_1 + beta_2 * x_2 + ss.norm.rvs(loc= 0, scale= 1, size= n)

X = np.stack([x_1, x_2], axis= 1)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection= '3d')
ax.scatter(X[:,0], X[:,1],y, c= y)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$y_1$")
ax.set_zlabel("$z_1$")
#plt.show()



# ============= LINEAR REGRESSION ==============
from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept= True)
lm.fit(X, y)
LinearRegression(copy_X= True, fit_intercept= True, n_jobs= 1, normalize= False)
print("Intercept:",lm.intercept_) #beta 0
print("Intercept:",lm.coef_) # beta1 and beta2

X_0 = np.array([2, 4])

#print(lm.predict(X_0.reshape(1,-1)))
#print(lm.score(X,y)) # Takes the y and compares the score with the rest of training data


# ============== SPLITTING DATA =================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size= 0.5, random_state= 1)
lm = LinearRegression(fit_intercept= True)
lm.fit(X_train, y_train)
print(lm.score(X_test, y_test))