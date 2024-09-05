import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

data_url = 'https://raw.githubusercontent.com/DilonSok/CS-4375-Intro-to-ML/main/Dilon-Sok%20HW1/student-por.csv'
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

X = data.drop('G3', axis=1)  #our features except the target
y = data['G3']  #our output target is G3 (end/overall grade)dadas

#split training/test data 80%/20% split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#(hyperparameters) all the learning rates and max iterations
learning_rates = [1.0, 0.1, 0.01, 0.0001, 0.00001, 0.000001]
max_iters = [100, 1000, 10000, 100000, 1000000]

#tracking variables for best variation of hyperparameters
best_learning_rate = 0
best_max_iter = 0
best_rmse = float('inf')
best_r2 = -float('inf')
best_weights = None

#tracking variables for worst variation of hyperparameters
worst_learning_rate = 0
worst_max_iter = 0
worst_rmse = 0
worst_r2 = float('inf')
worst_weights = None

# need to find the best hyperparameters
# test all variations of learning rates and max iterations
for curr_rate in learning_rates:
    for curr_iter in max_iters:

        #building the model
        model = SGDRegressor(max_iter=curr_iter, learning_rate="constant", eta0=curr_rate) 
        model.fit(X_train, y_train)

        #testing the models (generate Y predictions from X_test)
        y_test_predict = model.predict(X_test)

        #compute metrics (RMSE and R^2)
        rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
        r2 = r2_score(y_test, y_test_predict)

        #track best hyperparameters
        if rmse < best_rmse:
            best_learning_rate = curr_rate
            best_max_iter = curr_iter
            best_rmse = rmse
            best_r2 = r2
            best_weights = model.coef_

        #track worst hyperparameters
        if rmse > worst_rmse:
            worst_learning_rate = curr_rate
            worst_max_iter = curr_iter
            worst_rmse = rmse
            worst_r2 = r2
            worst_weights = model.coef_
        
        #printing the current variation 
        print(f"---Learning Rate: {curr_rate:<12.6f} Max_Iter: {curr_iter:<10} RMSE: {rmse:<10.6f} R2: {r2:<10.6f}---")

    print("\tBest is currently:")
    print(f"\t(Learning Rate: {best_learning_rate:.6f} Max_Iter: {best_max_iter:.6f} RMSE: {best_rmse:.6f} R2: {best_r2:.6f})\n")

#neatly print out weights coefs (best vs worst)
def print_weights(feature_names, best_weights, worst_weights):
    print(f"{'Feature':<20} {'Best Weight':<15} {'Worst Weight':<15}")
    print("-" * 50)
    for name, best_weight, worst_weight in zip(feature_names, best_weights, worst_weights):
        print(f"{name:<20} {best_weight:<15.6f} {worst_weight:<15.6f}")

#print out best vs worst variations and their metrics
print(f"{'Metric':<20} {'Best':<25} {'Worst':<25}")
print("-" * 70)
print(f"{'Learning Rate':<20} {best_learning_rate:<25.6f} {worst_learning_rate:<25.6f}")
print(f"{'Max Iterations':<20} {best_max_iter:<25} {worst_max_iter:<25}")
print(f"{'RMSE':<20} {best_rmse:<25.6f} {worst_rmse:<25.6f}")
print(f"{'R2':<20} {best_r2:<25.6f} {worst_r2:<25.6f}")

print("\nBest vs Worst Weights:")
print_weights(X.columns, best_weights, worst_weights)


#do graphs next




