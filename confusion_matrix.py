###This code shows fitting logistic model for a data set "heart.csv" using scikit learn and then implementation of confusion matrix and 
###plotting of roc and pr curve from scratch.

import pandas as pd
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, cut_tree
from ISLP.cluster import compute_linkage
from sklearn.decomposition import PCA

###QUES) Build a logistic regression model for the heart.csv and plot the roc and pr curves without using scikit-learn methods.

###loading the file
df = pd.read_csv("/home/ibab/movedfiles/heart.csv")
#print(df)

##dividing the data into features and target.
x = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

##splitting the data into test training set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

##fitting the model
model = LogisticRegression(max_iter=10000).fit(x_train, y_train)

###getting the probaility score after fitting the model and then dividing it into positive class and negative class.
y_pred_probability = model.predict_proba(x_test)
positive_class_prob = y_pred_probability[:, 1]
negative_class_prob = y_pred_probability[:, 0]
# print("y predicted probabilities of x_test are: ", "\n", y_pred_probability)

###varying thresholds to get different sensitivity, specificity and precision values to biuld roc and pr curve.
upper_half = []
lower_half = []
threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
tpr = []
fpr = []
Precision = []
for m in threshold:
    n = len(positive_class_prob)
    upper = int(n * m)
    upper_half = positive_class_prob[0:upper]
    lower_half = positive_class_prob[upper:len(positive_class_prob)]
    upper_half = np.array(upper_half)
    lower_half = np.array(lower_half)
    # print(upper_half)
    # print(lower_half)
    ##calculating true positive,false positive,false negative and true negative values.
    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0
    for k in range(len(upper_half)):
        if upper_half[k] > m:
            tp = tp + 1
        if upper_half[k] < m:
            fp = fp + 1
    for k in range(len(lower_half)):
        if lower_half[k] > m:
            fn = fn + 1
        if lower_half[k] < m:
            tn = tn + 1
    # print(tp, fp, fn, tn)
    ## recall is also called sensitivity
    ##roc: sensitivity(recall/tpr) against fpr(1-specificity)
    ##pr: precision against recall(sensitivity/tpr)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    precision = tp/(tp+fp)
    tpr.append(sensitivity)
    fpr.append(1-specificity)
    Precision.append(precision)

###plotting roc curve 
plt.plot(tpr, fpr)
plt.xlabel("TPR")
plt.ylabel("FPR")
plt.title("ROC curve")
plt.show()

###plotting pr curve
plt.plot(tpr, Precision)
plt.xlabel("TPR")
plt.ylabel("PRECISION")
plt.title("PR curve")
plt.show()

###printing model's accuracy.
y_pred = model.predict(x_test)
acc = accuracy_score(y_pred, y_test)
print("Accuracy is: ", acc)

