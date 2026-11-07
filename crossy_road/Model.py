import numpy as np
learningRate = 0.01
gamma = 0.1
done = 0

def bellman(r,q):
    Q = r + gamma * max(q) * (1 - done)
    return Q


inputs = np.array([])
IL = len(inputs)
layers = 5
neurons = 8
weights = np.random.randn(layers,neurons,IL)
biases = np.zeros(neurons)*layers

def model(A):
    for l in range(layers):
        Z = weights[l] @ A + biases[l]
        A = np.maximum(0,Z)
    return A

for l in range(layers):
weights[l] -= learningRate * dw[l]