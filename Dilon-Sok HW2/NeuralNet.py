import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import itertools 

data_url = 'https://raw.githubusercontent.com/DilonSok/CS-4375-Intro-to-ML/refs/heads/main/Dilon-Sok%20HW2/car-dataset/car.data'
column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
data = pd.read_csv(data_url, header=None, names=column_names)

class NeuralNet:
    def __init__(self, data):
        self.raw_input = data
        self.processed_data = None
        self.best_model = None
        self.best_accuracy = 0
        self.best_params = {}

    def preprocess(self):
        # Handle missing values
        self.raw_input = self.raw_input.dropna()

        features = self.raw_input.drop('class', axis=1)
        target = self.raw_input['class']

        #convert categorical features to numerical values
        label_encoders = {}
        for column in features.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            features[column] = le.fit_transform(features[column])
            label_encoders[column] = le

        target_encoder = LabelEncoder()
        target = target_encoder.fit_transform(target)

        #standardize features
        numerical_features = features.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        features[numerical_features] = scaler.fit_transform(features[numerical_features])

        #set processe data
        self.processed_data = pd.concat([features, pd.DataFrame(target, columns=['class'])], axis=1)

    def train_evaluate(self):
        # Separate features and target from preprocessed data
        ncols = len(self.processed_data.columns)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols - 1)]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Below are the hyperparameters that you need to use for model evaluation
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200]  # also known as epochs
        num_hidden_layers = [2, 3]

        #metric to track hyperparameters and model performance
        model_performance = []
        
        # Dictionary to store the history of loss curves for each activation function
        loss_history = {'logistic': [], 'tanh': [], 'relu': []}

        plt.figure(figsize=(16, 8))

        #loop through all hyperparamter combinations
        combination_count = 0
        for activation, lr, max_iter, layers in itertools.product(activations, learning_rate, max_iterations, num_hidden_layers):
            
            #assuming usage of 10 neurons per layer
            hidden_layer_size = tuple([10] * layers)
            
            #the model we are using from sklearn.neural_network
            model = MLPClassifier(hidden_layer_sizes=hidden_layer_size, activation=activation,
                                  learning_rate_init=lr, max_iter=max_iter, random_state=42)

            #train the model and store its training history
            history = model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            #metric data to collect
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            train_error = mean_squared_error(y_train, y_train_pred)
            test_error = mean_squared_error(y_test, y_test_pred)

            # Save performance metrics
            model_performance.append({
                'activation': activation,
                'learning_rate': lr,
                'epochs': max_iter,
                'hidden_layers': layers,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_error': train_error,
                'test_error': test_error
            })

            #tracking best model based on best test accuracy
            if test_accuracy > self.best_accuracy:
                self.best_accuracy = test_accuracy
                self.best_model = model
                self.best_params = {
                    'activation': activation,
                    'learning_rate': lr,
                    'epochs': max_iter,
                    'hidden_layers': layers,
                    'test_accuracy': test_accuracy
                }

             # Store the loss curve history for the corresponding activation function
            loss_history[activation].append((history.loss_curve_, f'LR: {lr}, Ep: {max_iter}, L: {layers}'))

            #summary for each combination
            print(f"Model {combination_count + 1}:")
            print(f"Activation: {activation}, Learning Rate: {lr}, Epochs: {max_iter}, Hidden Layers: {layers}")
            print(f"Training Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
            print(f"Training Error: {train_error:.4f}, Test Error: {test_error:.4f}\n")
            combination_count += 1

        #print table format for model performance
        results_df = pd.DataFrame(model_performance)
        print("\nSummary of Model Performance:")
        print(results_df)

        #print best model combination
        print("\nBest Model Combination:")
        print(f"Activation: {self.best_params['activation']}, Learning Rate: {self.best_params['learning_rate']}, "
              f"Epochs: {self.best_params['epochs']}, Hidden Layers: {self.best_params['hidden_layers']}")
        print(f"Best Test Accuracy: {self.best_params['test_accuracy']:.4f}")

        # Create separate plots for each activation function
        for activation in activations:
            plt.close('all')  #had to add this to prevent random extra blank figure window from showing up
            plt.figure(figsize=(12, 6)) 
            marker = itertools.cycle(('o', 'v', '^', '<', '>', 's', 'p', '*'))
            color = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'y', 'k')) 
            for loss_curve, label in loss_history[activation]:
                plt.plot(loss_curve, marker=next(marker), color=next(color), label=label)
            plt.title(f"Loss Curve for Activation Function: {activation}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend(loc='upper right')
            plt.show()
            plt.close() #had to add this to prevent random extra blank figure window from showing up


if __name__ == "__main__":

    neural_network = NeuralNet(data)
    neural_network.preprocess()
    neural_network.train_evaluate()
