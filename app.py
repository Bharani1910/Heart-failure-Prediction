from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model, scaler = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Retrieve values robustly by key to ensure the correct features order for the ML scaler
        inputs = [float(request.form[str(i)]) for i in range(12)]
        inputs_scaled = scaler.transform([inputs])

        prediction = model.predict(inputs_scaled)[0]

        # Extract important values (from index locations: 0=Age, 4=Ejection Fraction, 7=Serum Creatinine, 8=Serum Sodium)
        age = inputs[0]
        ejection_fraction = inputs[4]
        serum_creatinine = inputs[7]
        serum_sodium = inputs[8]

        if prediction == 1:
            return jsonify({
                "risk": "High Risk",
                "message": "Patient is likely to die.",
                "recommendation": "Immediate medical evaluation and close monitoring are strongly recommended."
            })
        else:
            # Count moderate-risk indicators
            risk_score = 0

            if age >= 65:
                risk_score += 1

            if ejection_fraction < 40:
                risk_score += 1

            if serum_creatinine > 1.2:
                risk_score += 1

            if serum_sodium < 135:
                risk_score += 1

            if risk_score >= 2:
                return jsonify({
                    "risk": "Moderate Risk",
                    "message": "Patient is likely to survive.",
                    "recommendation": "Regular medical monitoring, medication adherence, and follow-up are required."
                })
            else:
                return jsonify({
                    "risk": "Low Risk",
                    "message": "Patient is likely to survive.",
                    "recommendation": "Continue regular treatment and routine medical follow-up."
                })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
