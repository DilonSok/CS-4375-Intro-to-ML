import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

#retrieving the data and reading in from csv to data variable
data_url = 'https://raw.githubusercontent.com/DilonSok/CS-4375-Intro-to-ML/main/Dilon-Sok%20HW1/student-por.csv'
data = pd.read_csv(data_url, delimiter=";")
data = data.dropna() #dropping any rows that have null values

#encoding/converting categorical to numerical
label_encoder = LabelEncoder()
categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])
data = pd.get_dummies(data, columns=['Mjob', 'Fjob', 'reason', 'guardian'], prefix=['Mjob', 'Fjob', 'reason', 'guardian'])

#correlation matrix for our output variable G3 (the students final grade for the class)
correlation_with_g3 = data.corr()['G3'].sort_values(ascending=False)

#correlation heatmap graph
plt.figure(figsize=(6, 10))
sns.heatmap(correlation_with_g3.to_frame(), annot=True, fmt='.2f', cmap='coolwarm', cbar=False, center=0)
plt.title('Correlation of Features with Final Grade (G3)')
plt.show()

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
best_mean_cv_mse = 0
best_test_mse = float('inf') #track how well the model performs on the testing data
best_train_mse = float('inf') #track how well the model fits the training data
best_r2 = -float('inf')
best_weights = None

#for cross validation folds
k = 5 

#need to find the best hyperparameters
#test all variations of learning rates and max iterations
for curr_rate in learning_rates:
    for curr_iter in max_iters:

        #building the model
        model = SGDRegressor(max_iter=curr_iter, learning_rate="constant", eta0=curr_rate) 
        model.fit(X_train, y_train)

        cv_mse_scores = cross_val_score(model, X_train, y_train, cv=k, scoring='neg_mean_squared_error')
        mean_cv_mse = -np.mean(cv_mse_scores)

        y_train_predict = model.predict(X_train)
        y_test_predict = model.predict(X_test)

        train_mse = mean_squared_error(y_train, y_train_predict)
        test_mse = mean_squared_error(y_test, y_test_predict)
        r2 = r2_score(y_test, y_test_predict)

        #track best hyperparameters
        if test_mse < best_test_mse:
            best_learning_rate = curr_rate
            best_max_iter = curr_iter
            best_train_mse = train_mse
            best_test_mse = test_mse
            best_r2 = r2
            best_weights = model.coef_
            best_mean_cv_mse = mean_cv_mse
        
        #printing the current variation 
        print(f"Learning Rate: {curr_rate:<12.6f} | Max_Iter: {curr_iter:<10} | TRAIN_MSE: {train_mse:<10.6f} | TEST_MSE: {test_mse:<10.6f} | R2: {r2:<10.6f} | CV_MSE: {mean_cv_mse:.6f}")

    #print the current best metrics after testing out a learning rate to all max iteration variations
    print("\tBest is currently:")
    print(f"\t(Learning Rate: {best_learning_rate:.6f} | Max_Iter: {best_max_iter:.6f} | TRAIN_MSE: {best_train_mse:<10.6f} | TEST_MSE: {best_test_mse:.6f} | R2: {best_r2:.6f} | CV_MSE: {best_mean_cv_mse:.6f})\n")

#print out best variation and its metrics
print(f"{'Metric':<20} {'Best':<25}")
print("-" * 50)
print(f"{'Learning Rate':<20} {best_learning_rate:<25.6f}")
print(f"{'Max Iterations':<20} {best_max_iter:<25}")
print(f"{'TEST_MSE':<20} {best_test_mse:<25.6f}")
print(f"{'TRAIN_MSE':<20} {best_train_mse:<25.6f}")
print(f"{'MEAN_CV_MSE':<20} {best_mean_cv_mse:<25.6f}")
print(f"{'R2':<20} {best_r2:<25.6f}")

#use the best hyperparameters found and redo the model to get plotting data
model = SGDRegressor(learning_rate="constant", eta0=best_learning_rate)
train_mse_list = []
test_mse_list = []
cv_mse_list = []
iteration_checkpoints = []

#using partial fit to see MSEs incrementally 
for i in range(1000, best_max_iter + 1, 1000):
    model.max_iter = i
    model.partial_fit(X_train, y_train)
    
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_train_predict)
    test_mse = mean_squared_error(y_test, y_test_predict)
    
    cv_mse_scores = cross_val_score(model, X_train, y_train, cv=k, scoring='neg_mean_squared_error')
    mean_cv_mse = -np.mean(cv_mse_scores)
    
    train_mse_list.append(train_mse)
    test_mse_list.append(test_mse)
    cv_mse_list.append(mean_cv_mse)
    iteration_checkpoints.append(i)

#MSEs vs iterations
plt.figure(figsize=(10, 6))
plt.plot(iteration_checkpoints, train_mse_list, label='Train MSE', color='blue', marker='o')
plt.plot(iteration_checkpoints, test_mse_list, label='Test MSE', color='red', marker='x')
plt.plot(iteration_checkpoints, cv_mse_list, label='CV MSE', color='green', marker='s')
plt.xlabel('Number of Iterations')
plt.ylabel('MSE')
plt.title('MSE vs Number of Iterations (Best Hyperparameters)')
plt.legend()
plt.grid(True)
plt.show()

#output variable against some important features
important_features = ['G1', 'G2', 'studytime', 'failures', 'absences', 'higher']

plt.figure(figsize=(12, 8))
for i, feature in enumerate(important_features):
    plt.subplot(3, 2, i + 1) 
    plt.scatter(data[feature], y, color='green')
    plt.xlabel(feature)
    plt.ylabel('G3 (Final Grade)')
    plt.title(f'Final Grade vs {feature}')
plt.tight_layout()
plt.show()

#printing out best weights
print(f"\n{'Feature':<20} {'Best Weight':<15}")
print("-" * 35)
for name, best_weight in zip(X.columns, best_weights):
    print(f"{name:<20} {best_weight:<15.6f}")


