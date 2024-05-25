import random
import numpy as np  
from scipy import stats  

N = 100 # sample size
mu1 = mu2 = 10
num_tests = 10000

FPs = 0 # false positives
for _ in range(num_tests):
  # Gaussian distributed data with mean muX and var = 1  
  x = np.random.randn(N) + mu1
  y = np.random.randn(N) + mu2
  
  _t_stat, p_val = stats.ttest_ind(x, y)
  if p_val < 0.05:
    FPs += 1

print(f'False Positive Rate: {FPs / num_tests:.2f}')

FPs = 0
for _ in range(num_tests):
  x = np.random.randn(N) + mu1
  y = np.random.randn(N) + mu2
  
  x = [random.random() * x + 5 for x in x]
  y = [random.random() * y + 5 for y in y]
  
  _t_stat, p_val = stats.ttest_ind(x, y)
  if p_val < 0.05:
    FPs += 1

print(f'False Positive Rate: {FPs / num_tests:.2f}')

FPs = 0
for _ in range(num_tests):
  x = np.random.randn(N) + mu1
  y = np.random.randn(N) + mu2
  
  x = [random.random() ** x + 5 for x in x]
  y = [random.random() ** y + 5 for y in y]
  
  _t_stat, p_val = stats.ttest_ind(x, y)
  if p_val < 0.05:
    FPs += 1

print(f'False Positive Rate: {FPs / num_tests:.2f}')

FPs = 0
for _ in range(num_tests):
  x = np.random.randn(N) + mu1
  y = np.random.randn(N) + mu2
  
  x = [x ** (random.random() * 2) + 5 for x in x]
  y = [y ** (random.random() * 2) + 5 for y in y]
  
  _t_stat, p_val = stats.ttest_ind(x, y)
  if p_val < 0.05:
    FPs += 1

print(f'False Positive Rate: {FPs / num_tests:.2f}')
