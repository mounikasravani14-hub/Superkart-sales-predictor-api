
# Import necessary libraries
import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask application
Superkart_sales_predictor_api = Flask("Superkart sales predictor")

# Load the trained machine learning model
model = joblib.load("Product_Store_Sales_Total.joblib")


# --------------------------------------------------
# Home page
# --------------------------------------------------

@Superkart_sales_predictor_api.route("/")
def home():
    return "<h1>SuperKart Sales Predictor</h1>"


# --------------------------------------------------
# Single Prediction
# --------------------------------------------------

@Superkart_sales_predictor_api.route("/predict", methods=["POST"])
def predict():

    # Get JSON data from request
    data = request.get_json()

    # Extract relevant features
    sample = {
        "Product_Weight": data["Product_Weight"],
        "Product_Sugar_Content": data["Product_Sugar_Content"],
        "Product_Allocated_Area": data["Product_Allocated_Area"],
        "Product_MRP": data["Product_MRP"],
        "Store_Size": data["Store_Size"],
        "Store_Location_City_Type": data["Store_Location_City_Type"],
        "Store_Type": data["Store_Type"],
        "Store_Age": data["Store_Age"]
    }

    # Convert extracted data into DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction
    prediction = model.predict(input_data)

    # Return prediction
    return jsonify({
        "prediction": float(prediction[0])
    })


# --------------------------------------------------
# Batch Prediction
# --------------------------------------------------

@Superkart_sales_predictor_api.route("/predict_batch", methods=["POST"])
def predict_batch():

    # Get JSON data from request
    data = request.get_json()

    # Convert JSON data into DataFrame
    batch_data = pd.DataFrame(data)

    # Fix inconsistent category
    batch_data["Product_Sugar_Content"] = (
        batch_data["Product_Sugar_Content"]
        .replace({"reg": "Regular"})
    )

    # Create Store_Age from Store_Age_Years
    batch_data["Store_Age"] = batch_data["Store_Age_Years"]

    # Select only features used by the trained model
    features = [
        "Product_Weight",
        "Product_Sugar_Content",
        "Product_Allocated_Area",
        "Product_MRP",
        "Store_Size",
        "Store_Location_City_Type",
        "Store_Type",
        "Store_Age"
    ]

    batch_data = batch_data[features]

    # Make predictions
    predictions = model.predict(batch_data)

    # Return predictions as JSON
    return jsonify({
        "predictions": predictions.tolist()
    })


# --------------------------------------------------
# Run Flask application
# --------------------------------------------------

if __name__ == "__main__":
    Superkart_sales_predictor_api.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
