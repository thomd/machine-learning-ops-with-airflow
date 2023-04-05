from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import pickle

def download_dataset():
    iris = load_iris()
    iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
    pd.DataFrame(iris).to_csv('iris_dataset.csv')

def data_processing():
    data = pd.read_csv('iris_dataset.csv', index_col=0)
    cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    data[cols] = data[cols].fillna(data[cols].mean())
    data.to_csv('iris_dataset.clean.csv')

def ml_training_randomforest(**kwargs):
    data = pd.read_csv('iris_dataset.clean.csv', index_col=0)
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,0:4], data.iloc[:,-1], test_size=0.3)
    rf_classifier = RandomForestClassifier(n_estimators=100)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'accuracy random-forrest: {acc}')
    kwargs['ti'].xcom_push(key='model_accuracy', value=acc)
    pickle.dump(rf_classifier, open(f'model_randomforrest_{acc}.pickle', 'wb'))

def ml_training_logistic(**kwargs):
    data = pd.read_csv('iris_dataset.clean.csv', index_col=0)
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,0:4], data.iloc[:,-1], test_size=0.3)
    lr_classifier = LogisticRegression(multi_class='ovr')
    lr_classifier.fit(X_train, y_train)
    y_pred = lr_classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'accuracy logistic regression: {acc}')
    kwargs['ti'].xcom_push(key='model_accuracy', value=acc)
    pickle.dump(lr_classifier, open(f'model_logisticregression_{acc}.pickle', 'wb'))

def identify_best_model(**kwargs):
    ti = kwargs['ti']
    rf_acc = ti.xcom_pull(key='model_accuracy', task_ids=['train_randomforest'])
    lr_acc = ti.xcom_pull(key='model_accuracy', task_ids=['train_logistic'])
    with open('accuracy.txt', 'w') as f:
        f.write(f'random forest accuracy: {rf_acc[0]}\n')
        f.write(f'logistic regression accuracy: {lr_acc[0]}')
