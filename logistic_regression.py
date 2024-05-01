###Implement logistic regression using scikit-learn for the breast cancer dataset and simulated dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
def load_data():
    [X, y] = load_breast_cancer(return_X_y=True)
    #print( [X, y])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, shuffle=False)
    #print(X_train, X_test, y_train, y_test)
    logreg = LogisticRegression(max_iter=10000).fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(acc)
def main():
    load_data()
if __name__=="__main__":
    main()

