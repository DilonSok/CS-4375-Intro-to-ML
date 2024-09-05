import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import SGDRegressor

data_url = 'https://raw.githubusercontent.com/DilonSok/ML-Datasets/main/student-por.csv'
data = pd.read_csv(data_url, delimiter=";")
data = data.dropna()

data['school'] = data['school'].map({'GP': 0, 'MS': 1})
data['sex'] = data['sex'].map({'F': 0, 'M': 1})
data['address'] = data['address'].map({'U': 0, 'R': 1})
data['famsize'] = data['famsize'].map({'LE3': 0, 'GT3': 1})
data['Pstatus'] = data['Pstatus'].map({'T': 0, 'A': 1})
data = pd.get_dummies(data, columns=['Mjob', 'Fjob', 'reason', 'guardian'], prefix=['Mjob', 'Fjob', 'reason', 'guardian'])
data['schoolsup'] = data['schoolsup'].map({'no': 0, 'yes': 1})
data['famsup'] = data['famsup'].map({'no': 0, 'yes': 1})
data['paid'] = data['paid'].map({'no': 0, 'yes': 1})
data['activities'] = data['activities'].map({'no': 0, 'yes': 1})
data['nursery'] = data['nursery'].map({'no': 0, 'yes': 1})
data['higher'] = data['higher'].map({'no': 0, 'yes': 1})
data['internet'] = data['internet'].map({'no': 0, 'yes': 1})
data['romantic'] = data['romantic'].map({'no': 0, 'yes': 1})

X = data.drop('G3', axis=1)  # Features
y = data['G3']  # Target is G3 (end/overall grade)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

learning_rates = [0.1, 0.01, 0.0001, 0.00001, 0.000001]

for rate in learning_rates:
    model = SGDRegressor(alpha=rate, max_iter=1000) # need to find the best hyperparameters
    model.fit(X_train, y_train)
    y_test_predict = model.predict(X_test)
    rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
    r2 = r2_score(y_test, y_test_predict)
    print(r2)





