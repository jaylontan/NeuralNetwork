import numpy as np
from activations import *
from main import MyNeuralNetwork

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

n = 1000  # Number of runs
accuracies = []

for i in range(n):
    # Reinitialize and train a new model each time
    nn = MyNeuralNetwork()
    nn.Linear(2, 3)
    nn.ReLU()
    nn.Linear(3, 6)
    nn.ReLU()
    nn.Linear(6, 1)
    nn.Sigmoid()
    
    nn.train(X, y, num_of_epochs=10000)

    # Evaluate accuracy
    y_pred = nn.forward(X)
    y_pred_class = (y_pred > 0.5).astype(int)
    acc = np.mean(y_pred_class == y.reshape(-1, 1))
    accuracies.append(acc)

# Print average accuracy
avg_acc = np.mean(accuracies)
print(f"\nAverage Accuracy over {n} runs: {avg_acc * 100:.2f}%")

