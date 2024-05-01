import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np
from matplotlib import pyplot as plt

## Loading data
df = pd.read_csv("/home/ibab/movedfiles/heart.csv")
#print(df)

##splitting data into features(x) and target(y)
x = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

##constructing correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

## plotting box plots(to examine outliers) and histogram for all the features to analyze the data well.
for i in x:
    plt.boxplot(x[i])
    plt.show()
    plt.xlabel("feature")
    plt.ylabel("frequency")
    plt.hist(x[i])
    plt.show()
    plt.xlabel("feature")
    plt.ylabel("frequency")
acc_sc = []
k = 10
kf = KFold(n_splits=k, random_state=42, shuffle=True)
for train_index, test_index in kf.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    ##encoding is not required as the data set does not have string values in any of the features.
    ## standardizing the splitted data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    ## building and predicting different models for each k-fold

    model = LogisticRegression()
    #model = RandomForestClassifier()
    #model = xgb.XGBClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    acc_sc.append(score)
#print(acc_sc)
mean = np.mean(acc_sc)
print("Mean: ", mean)
std = np.std(acc_sc)
print("Standard deviation: ", std)

## printing a comparison of mean and standard deviation for all three models.
comparison = ["Logistic_Regression: mean->0.8429754426042262, std->0.04486893704150521", "Randon_Forest: mean->0.9970873786407767, std->0.0087378640776699", "XGBoost mean->0.9970873786407767, std->0.0087378640776699"]
print("----------------------------------------")
print("RESULTS")
print(comparison)
