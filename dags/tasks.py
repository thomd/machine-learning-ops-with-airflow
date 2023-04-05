from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

def download_dataset():
    iris = load_iris()
    iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
    pd.DataFrame(iris).to_csv('iris_dataset.csv')

def data_processing():
    final = pd.read_csv('iris_dataset.csv', index_col=0)
    cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    final[cols] = final[cols].fillna(final[cols].mean())
    final.to_csv('iris_dataset.clean.csv')

def ml_training_RandomForest(**kwargs):
    final = pd.read_csv('iris_dataset.clean.csv', index_col=0)
    X_train, X_test, y_train, y_test = train_test_split(final.iloc[:,0:4], final.iloc[:,-1], test_size=0.3)
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc}')
    kwargs['ti'].xcom_push(key='model_accuracy', value=acc)

def ml_training_Logisitic(**kwargs):
    final = pd.read_csv('iris_dataset.clean.csv', index_col=0)
    X_train, X_test, y_train, y_test = train_test_split(final.iloc[:,0:4], final.iloc[:,-1], test_size=0.3)
    logistic_regression = LogisticRegression(multi_class='ovr')
    lr = logistic_regression.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc}')
    kwargs['ti'].xcom_push(key='model_accuracy', value=acc)

def identify_best_model(**kwargs):
    ti = kwargs['ti']
    fetched_accuracies = ti.xcom_pull(key='model_accuracy', task_ids=['ml_training_randomforest', 'ml_training_logisitic'])
    print(f'best model: {fetched_accuracies}')
