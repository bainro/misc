import numpy as np  
from scipy import stats  

N = 30            # sample size
mu1 = mu2 = 10
FPs = 0           # false positives
num_tests = 1000

for _ in range(num_tests):
  # Gaussian distributed data with mean = 10.5 and var = 1  
  x = np.random.randn(N) + mu1
  y = np.random.randn(N) + mu2
  
  ## Using the internal function from SciPy Package  
  _t_stat, p_val = stats.ttest_ind(x, y)
  print(f'p-value: {p_val}')
  if p_val < 0.05:
    FPs += 1

print(f'False Positive Rate: {FPs / num_tests:.2f}')
