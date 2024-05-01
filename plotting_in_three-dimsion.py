from matplotlib import pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
from numpy.linalg import norm
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

##Plot these points.  Then transform these points using your “Transform” function into 3-dim space. Plot the points and
## manipulate the points so that you can see a separating plane in 3D.

def transform(X1, root, X2, label):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1, root, X2, c=label)
    plt.show()
def threeD(x1,x2, label):
    X1 = []
    root = []
    X2 = []
    for i in range(len(x1)):
        cal1 = x1[i]**2
        X1.append(cal1)
    for j in range(len(x2)):
        cal2 = x2[j]**2
        X2.append(cal2)
    for k in range(len(x1)):
        cal3 = 2**0.5*x1[k]*x2[k]
        root.append(cal3)
    transform(X1, root, X2, label)
def phi(x1,x2,label):
    plt.scatter(x1, x2, c=label)
    plt.show()
    threeD(x1, x2, label)
def main():
    x1 = [1, 1, 2, 3, 6, 9, 13, 18, 13, 18, 9, 6, 3, 2, 1, 1]
    x2 = [3, 6, 6, 9, 10, 11, 12, 16, 15, 6, 11, 5, 10, 5, 6, 3]
    label = ['Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Red', 'Red', 'Red', 'Red', 'Red', 'Red', 'Red', 'Red']
    phi(x1, x2, label)
if __name__=="__main__":
    main()

