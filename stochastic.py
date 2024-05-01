##stochastic gradient

import numpy as np
from sklearn.metrics import r2_score
import pandas as pd

df = pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")

insert = df.insert(0,'x_0',1)
#print(df)

n = df.shape[0]   #rows
d = df.shape[1]   #cols
#print(n,d)

###division of data
X = df.iloc[4, 0:d-2]
y = df.iloc[4, d-2]
#print(X, y)

theta = np.zeros(6)
h_x = 0
###hypothesis function
h_x += theta*X
#print(h_x)
cost_func = sum(0.5*(h_x - y)**2)
#print(cost_func)
alpha = 0.0000001
threshold = 0.01
a = 100000
derivative = (h_x - y) * X
#print(derivative)
while a > threshold:
      theta = theta - alpha * derivative
      h_x += theta * X
      cost_func_new = sum(0.5 * (h_x - y) ** 2)
      print(cost_func_new)
      a = cost_func - cost_func_new
      if a < threshold:
         break
      cost_func = cost_func_new
print(theta)

