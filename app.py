from flask import Flask, request, jsonify
from flask.logging import create_logger
import logging
import pandas as pd
import mlib

app = Flask(__name__)
LOG = create_logger(app)
LOG.setLevel(logging.INFO)


model_name = "Predict if a user is default"
version = "v1.0.0"


@app.route("/")
def home():
    """Return model information, version, how to call"""
    result = {}

    result["name"] = model_name
    result["version"] = version

    return result


@app.route("/predict", methods=["POST"])
def predict():
    """Predict the user is a default"""

    json_payload = request.json
    LOG.info(f"JSON payload: {json_payload}")
    df = pd.DataFrame(json_payload)
    prediction = mlib.predict(df)
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
