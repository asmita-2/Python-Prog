### Implementing batch-gradient descent from scratch for the dataset simulated_data_multiple_linear_regression_for_ML.csv. This code shows how the cost
### function comes down and at convergenge prints all the parameters(theta values).

###importing necessary libraries
import numpy as np
import pandas as pd

###loading the data
df = pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")

###inserting a column in the data containing value 1 in all the rows 
insert = df.insert(0,'x_0',1)
#print(df)


n = df.shape[0]   #no. of rows
d = df.shape[1]   #no. of cols
#print(n,d)

###division of data into features and target.
X = df.iloc[:, 0:d-2]
y = df.iloc[:, d-2]
#print(X, y)

##initializing parameter(theta)values
theta = np.zeros(6)
h_x = 0

### defining hypothesis function
for i in range(0, d-2):
   h_x += theta[i]*X.iloc[:, i]
#print(h_x)

##definimg cost function and updating cost function
cost_func = sum(0.5*(h_x - y)**2)
print(cost_func)

alpha = 0.0000001
threshold = 0.0001
a = 100000
while a > threshold:
   for j in range(d-2):
      derivative = sum((h_x - y) * X.iloc[:, j])
      theta[j] = theta[j] - alpha * derivative
   h_x += theta[i] * X.iloc[:, i - 1]
   cost_func_new = sum(0.5 * (h_x - y) ** 2)
   print(cost_func_new)
   a = cost_func - cost_func_new
   if a < threshold:
         break
   cost_func = cost_func_new
   
##Lastly printing the theta values
print(theta)

