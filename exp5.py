from flask import Flask, render_template, request
import numpy as np
from tensorflow import keras

app = Flask(__name__, template_folder=".")

X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([[10], [20], [30], [40], [50]], dtype=float)

model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(1, kernel_initializer="zeros",
                       bias_initializer="zeros")
])

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    loss="mse"
)
model.fit(X, y, epochs=2000, verbose=0)

@app.route("/", methods=["GET", "POST"])
def home():
    price = None

    if request.method == "POST":
        size = float(request.form["size"])
        result = model.predict(np.array([[size]]), verbose=0)
        price = round(float(result[0][0]), 2)

    return render_template("index.html", price=price)

if __name__ == "__main__":
    app.run()
