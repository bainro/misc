import numpy as np  
from scipy import stats  

# Sample Sizes
N = 30
mu1 = mu2 = 10

# Gaussian distributed data with mean = 10.5 and var = 1  
x = np.random.randn(N) + mu1
y = np.random.randn(N) + mu2

## Using the internal function from SciPy Package  
t_stat, p_val = stats.ttest_ind(x, y)
print("t-statistic = " + str(t_stat))
print("p-value = " + str(p_val))
