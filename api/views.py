import os
import numpy as np
import pickle
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and preprocessing components
with open(os.path.join(BASE_DIR, 'mdl.pkl'), "rb") as f:
    data = pickle.load(f)

# Extract components from the model data
le_soil = data["label_encoder_soil"]
le_month = data["label_encoder_month"]
le_crop = data["label_encoder_crop"]
scaler = data["scaler"]
model = data["model"]

# Home route
@api_view(['GET'])
def home(request):
    return Response({"message": "Welcome to the crop recommendation API!"})

# Prediction route
@api_view(['POST'])
def predict(request):
    data = request.data
    
    # Extract input data
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    soil_type = data.get("soil_type")
    temperature = data.get("temperature")
    humidity = data.get("humidity")
    month = data.get("month")

    # Check for missing inputs
    if None in [latitude, longitude, soil_type, temperature, humidity, month]:
        return Response({"error": "Missing input data"}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Encode soil type and month
        soil_type_encoded = le_soil.transform([soil_type])[0]
        month_encoded = le_month.transform([month])[0]
    except ValueError:
        return Response({"error": "Invalid soil_type or month"}, status=status.HTTP_400_BAD_REQUEST)

    # Prepare input data for prediction
    input_data = np.array([[latitude, longitude, soil_type_encoded, temperature, humidity, month_encoded]])
    input_data_scaled = scaler.transform(input_data)

    # Make predictions
    predictions = model.predict(input_data_scaled)
    predicted_class_index = np.argmax(predictions)

    # Decode the predicted class index to crop name
    predicted_crop_name = le_crop.inverse_transform([predicted_class_index])[0]

    return Response({"predicted_crop": predicted_crop_name})
