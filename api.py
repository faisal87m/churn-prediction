#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API for Customer Churn Prediction.
This API uses a pre-trained pipeline to make predictions.
"""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import uvicorn
import os

# --- Import project functions ---
# It is assumed this file is in the project root directory
from src.features import create_features

# --- 1. Load the model at server startup (most efficient way) ---
# The model is loaded only once when the server starts.
MODEL_PATH = "models/churn_model.pkl"
try:
    # The saved model is a complete Pipeline that includes preprocessing.
    model = joblib.load(MODEL_PATH)
    print(f"Model pipeline loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. Please run the training pipeline first."
    )

# --- 2. Create FastAPI app and data models ---
app = FastAPI(
    title="Customer Churn Prediction System (Pipeline-based)",
    description="API using a complete pipeline for churn prediction.",
    version="1.0",
)


# Data model for a single event (as expected by create_features)
class Event(BaseModel):
    userId: str  # Unique user identifier
    ts: int  # Timestamp of the event
    registration: int  # User registration timestamp
    page: str  # Page visited or action taken
    sessionId: int  # Session identifier
    gender: str  # User gender
    location: str  # User location
    level: str  # Subscription level (e.g., free, paid)


# Request model: accepts a list of events for one or more users.
class PredictionRequest(BaseModel):
    events: List[Event] = Field(
        ...,
        example=[
            {
                "userId": "123",
                "ts": 1538352011000,
                "registration": 1538340000000,
                "page": "NextSong",
                "sessionId": 1,
                "gender": "F",
                "location": "New York",
                "level": "paid",
            }
        ],
    )


# --- 3. Main prediction endpoint ---
@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Predicts the probability of churn for one or more users based on their event lists.
    """
    if not request.events:
        # No events provided in the request
        raise HTTPException(status_code=400, detail="No events provided.")

    try:
        # 1. Convert the request data (raw events) to a DataFrame
        raw_events_df = pd.DataFrame([event.dict() for event in request.events])

        # 2. Generate features using the custom function
        # This function will aggregate events for each user
        features_df = create_features(raw_events_df)

        if features_df.empty:
            # If no features could be generated, return an error
            raise HTTPException(
                status_code=400,
                detail="Could not generate features from the provided events.",
            )

        # 3. Predict using the full pipeline
        # No manual preprocessing needed, the pipeline handles everything
        # Drop columns not used during training
        features_to_predict = features_df.drop(
            columns=["userId", "is_churned"], errors="ignore"
        )

        probabilities = model.predict_proba(features_to_predict)

        # 4. Prepare results for each user
        results = []
        for index, user_id in enumerate(features_df["userId"]):
            churn_probability = probabilities[index][
                1
            ]  # Probability for class 1 (Churn)
            results.append(
                {
                    "userId": user_id,
                    "churn_probability": float(churn_probability),
                    "prediction": "Churn" if churn_probability >= 0.5 else "No Churn",
                }
            )

        return {"predictions": results}
    except Exception as e:
        # Handle any other errors
        raise HTTPException(
            status_code=500, detail=f"An error occurred during prediction: {str(e)}"
        )


if __name__ == "__main__":
    # Run the server if this file is executed directly
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
