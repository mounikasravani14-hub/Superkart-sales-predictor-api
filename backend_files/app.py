# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask
# Initialize the Flask application
Superkart_sales_predictor_api = Flask("Superkart sales predictor")

#load the trained machinelearning model
model =joblib.load("Product_Store_Sales_Total.joblib")

#define a route for the home page (GET request)
@Superkart_sales_predictor_api.route("/")
def home():
    return "<h1>SuperKart Sales Predictor</h1>"

# Define an endpoint for single property prediction (POST request)
@Superkart_sales_predictor_api.route("/predict", methods=["POST"])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

# Extract relevant features from the JSON
sample = {
    'Product_Weight': data['Product_Weight'],
    'Product_Sugar_Content': data['Product_Sugar_Content'],
    'Product_Allocated_Area': data['Product_Allocated_Area'],
    'Product_MRP': data['Product_MRP'],
    'Store_Size': data['Store_Size'],
    'Store_Location_City_Type': data['Store_Location_City_Type'],
    'Store_Type': data['Store_Type'],
    'Store_Age': data['Store_Age']
}


#convert the extracted data into a Pandas DataFrame
input_data = pd.DataFrame([sample])


# Make prediction
prediction = model.predict(input_data)

# Return prediction
return jsonify({
        "prediction": float(prediction[0])
    })


# Define an endpoint for batch prediction
@Superkart_sales_predictor_api.route("/predict_batch", methods=["POST"])
def predict_batch():

    # Get JSON data from the request
    data = request.get_json()

    # Convert JSON data into a DataFrame
    batch_data = pd.DataFrame(data)

    # Fix inconsistent category
    batch_data["Product_Sugar_Content"] = (
        batch_data["Product_Sugar_Content"]
        .replace({"reg": "Regular"})
    )

    # Create Store_Age from Store_Age_Years
    batch_data["Store_Age"] = batch_data["Store_Age_Years"]

    # Select only the features used by the trained model
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

    # Run the Flask application in debug mode if this script is executed directly
    if __name__ == "__main__":
        Superkart_sales_predictor_api.run(debug=True)
