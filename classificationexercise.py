import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
from joblib import dump, load
import numpy as np


def data_preprocessing(file):
    inputDF = pd.read_csv(file)

    inputDF['x41'] = inputDF['x41'].str.lstrip('$')
    inputDF['x45'] = inputDF['x45'].str.rstrip('%')

    inputDF['x34'].replace(np.nan, 'empty', inplace=True)
    inputDF['x35'].replace(np.nan, 'empty', inplace=True)
    inputDF['x68'].replace(np.nan, 'empty', inplace=True)
    inputDF['x93'].replace(np.nan, 'empty', inplace=True)

    inputDF.dropna(inplace=True)

    return inputDF

def data_preprocessing_test(file):
    inputDF = pd.read_csv(file)
    #inputDF.dropna(inplace=True)

    inputDF['x41'] = inputDF['x41'].str.lstrip('$')
    inputDF['x45'] = inputDF['x45'].str.rstrip('%')

    inputDF['x34'].replace(np.nan, 'empty', inplace=True)
    inputDF['x35'].replace(np.nan, 'empty', inplace=True)
    inputDF['x68'].replace(np.nan, 'empty', inplace=True)
    inputDF['x93'].replace(np.nan, 'empty', inplace=True)

    inputDF = inputDF.replace(np.nan, 0)

    return inputDF

def onehot_encoding(X, columns):
    labellec_x = LabelEncoder()
    for column in columns:
        X.iloc[:, column] = labellec_x.fit_transform(X.iloc[:, column])

    onehotencoder = OneHotEncoder(categorical_features=columns, n_values='auto',
                                  handle_unknown='ignore')
    X = onehotencoder.fit_transform(X)
    return X


def train_test_model(classifier,model_name, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accu_score = accuracy_score(y_test, y_pred)
    dump(classifier, str(model_name)+'.joblib')

    return accu_score

def predict_test(model_name, X, filename):

    classifier = load(str(model_name)+'.joblib')
    y_pred = classifier.predict_proba(X)
    y_pred = y_pred[:, 0]
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = ['0']
    y_pred.to_csv(filename,sep=',', encoding='utf-8')



inputDF = data_preprocessing('exercise_01_train.csv')
X = inputDF.iloc[:,:-1]
y = inputDF.iloc[:,-1]

X = onehot_encoding(X, [34,35,68,93])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)


# testing Logistic regression
accu_score_log_reg = train_test_model(LogisticRegression(random_state=24), 'log_reg',X_train, X_test, y_train, y_test)
print (accu_score_log_reg)

# testing Random Forest
accu_score_rand_forest = train_test_model(RandomForestClassifier(n_estimators=50), 'rand_forest',X_train, X_test, y_train, y_test)
print (accu_score_rand_forest)



# Final prediction using Logistic model
X = data_preprocessing_test('exercise_01_test.csv')
X = onehot_encoding(X, [34,35,68,93])


predict_test('log_reg', X, 'results1.csv')

predict_test('rand_forest', X,'results2.csv')