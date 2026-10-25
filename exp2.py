
import numpy as np
from tensorflow import keras

X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([[10], [20], [30], [40], [50]], dtype=float)

model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(1)
])

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    loss="mse"
)

model.fit(X, y, epochs=2000, verbose=0)

price = model.predict(np.array([[6.0]]), verbose=0)
print("Predicted price:", round(float(price[0][0]), 2), "lakhs")
