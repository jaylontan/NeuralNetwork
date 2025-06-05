import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from activations import *
from main import NeuralNetwork

X = np.array([
    [150, 70], 
    [254, 73],
    [312, 68],
    [120, 60],
    [154, 61],
    [212, 65],
    [216, 67],
    [145, 67],
    [184, 64],
    [130, 69]
])

X = (X - X.mean(axis=0)) / X.std(axis=0)


y = np.array([
    0,  
    1,   
    1, 
    0,
    0,
    1,
    1,
    0,
    1,
    0
])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

n = 10  # Number of runs
accuracies = []

for i in range(n):
    # Reinitialize and train a new model each time
    nn = NeuralNetwork()
    nn.Linear(2, 3, 'he')
    nn.ReLU()
    nn.Linear(3, 6, 'he')
    nn.ReLU()
    nn.Linear(6, 1, 'he')
    nn.Sigmoid()
    
    # nn.train_mini_batch(X, y, num_of_epochs=10000, learning_rate=0.1, batch_size=3, lambda_l=0.01, threshold=10000)
    ## nn.train(X_train, y_train, num_of_epochs=10000, learning_rate=0.01, lambda_l=0.01, threshold=15)
    nn.train_stochastic(X_train, y_train, num_of_epochs=10000, learning_rate=0.01, lambda_l=0.01,threshold=1000)

    # Evaluate accuracy
    y_pred = nn.forward(X_test)
    y_pred_class = (y_pred > 0.5).astype(int)
    acc = np.mean(y_pred_class == y_test.reshape(-1, 1))
    accuracies.append(acc)

# Print average accuracy
avg_acc = np.mean(accuracies)
print(f"\nAverage Accuracy over {n} runs: {avg_acc * 100:.2f}%")

# Add L2 Regularization