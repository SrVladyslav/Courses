import numpy as np 
import pandas as pd 
import scipy.stats as ss
import matplotlib.pyplot as plt 
import statsmodels.api as sm 

# Generating points
n = 100
beta_0 = 5
beta_1 = 2
np.random.seed(1)

# [0-10]
x = 10 * ss.uniform.rvs(size= n)
y = beta_0 + beta_1 * x + ss.norm.rvs(loc= 0, scale= 1, size= n) # ss.norm is noise

# MODEL
X = sm.add_constant(x)
mod = sm.OLS(y, X) # adding the model with constants
est = mod.fit()
print(est.summary())