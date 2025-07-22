from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model, scaler = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    inputs = [float(x) for x in request.form.values()]
    inputs_scaled = scaler.transform([inputs])
    prediction = model.predict(inputs_scaled)[0]
    result = "Patient likely to die 😢" if prediction == 1 else "Patient likely to survive 😊"
    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
