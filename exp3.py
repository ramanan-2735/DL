import numpy as np
from tensorflow import keras

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=3, verbose=0)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy:", round(accuracy * 100, 2), "%")

prediction = model.predict(X_test[:1], verbose=0)
print("Predicted digit:", np.argmax(prediction))
print("Actual digit:", y_test[0])
