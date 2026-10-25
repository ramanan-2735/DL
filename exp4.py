from tensorflow import keras

(X_train, _), (X_test, _) = keras.datasets.mnist.load_data()

X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

encoder = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(32, activation="relu")
])

decoder = keras.Sequential([
    keras.layers.Input(shape=(32,)),
    keras.layers.Dense(784, activation="sigmoid")
])

model = keras.Sequential([encoder, decoder])
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, X_train, epochs=3, verbose=0)

compressed = encoder.predict(X_test[:1], verbose=0)
reconstructed = model.predict(X_test[:1], verbose=0)

print("Original shape:", X_test[:1].shape)
print("Compressed shape:", compressed.shape)
print("Reconstructed shape:", reconstructed.shape)
