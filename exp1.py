import numpy as np

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(0)
w1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))
w2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

lr = 1

for i in range(10000):
    hidden = 1 / (1 + np.exp(-(X @ w1 + b1)))
    output = 1 / (1 + np.exp(-(hidden @ w2 + b2)))

    error = output - y
    hidden_error = (error @ w2.T) * hidden * (1 - hidden)

    w2 -= lr * (hidden.T @ error) / 4
    b2 -= lr * np.mean(error, axis=0, keepdims=True)
    w1 -= lr * (X.T @ hidden_error) / 4
    b1 -= lr * np.mean(hidden_error, axis=0, keepdims=True)

print("Predictions:")
print(np.round(output).astype(int))
