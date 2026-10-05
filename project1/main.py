import logging
import os
from contextlib import asynccontextmanager
from enum import Enum
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mental_health_api")

# Model path configuration
MODEL_FILE = os.path.join(os.path.dirname(__file__), "Mental_Health_Model.pkl")
model = None

# Top 10 countries as extracted during model training
TOP_COUNTRIES = {
    "Other",
    "India",
    "USA",
    "Canada",
    "Australia",
    "UK",
    "Germany",
    "Mexico",
    "Turkey",
    "France",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to load ML model on startup."""
    global model
    if os.path.exists(MODEL_FILE):
        try:
            logger.info(f"Loading model from {MODEL_FILE}...")
            model = joblib.load(MODEL_FILE)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model file: {e}")
            model = None
    else:
        logger.warning(f"Model file '{MODEL_FILE}' not found.")
    yield
    logger.info("Shutting down API service.")


# FastAPI app initialization
app = FastAPI(
    title="Student Mental Health Impact Prediction API",
    description=(
        "API for predicting student mental health impact scores based on "
        "social media usage, screen time, study habits, physical activity, "
        "and sleep patterns."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Enums for Categorical Inputs ---
class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"
    other = "Other"


class AcademicLevelEnum(str, Enum):
    undergraduate = "Undergraduate"
    graduate = "Graduate"
    high_school = "High School"


class PlatformEnum(str, Enum):
    instagram = "Instagram"
    facebook = "Facebook"
    whatsapp = "WhatsApp"
    linkedin = "LinkedIn"
    snapchat = "Snapchat"
    tiktok = "TikTok"
    youtube = "YouTube"
    twitter = "Twitter"
    vkontakte = "VKontakte"
    other = "Other"


class PurposeOfUseEnum(str, Enum):
    entertainment = "Entertainment"
    education = "Education"
    networking = "Networking"
    news = "News"


class StressLevelEnum(str, Enum):
    low = "Low"
    medium = "Medium"
    high = "High"
    very_high = "Very High"


# --- Pydantic Data Schemas ---
class StudentInput(BaseModel):
    Age: int = Field(..., ge=10, le=100, description="Age of the student (years)", example=21)
    Gender: GenderEnum = Field(..., description="Gender identity", example=GenderEnum.male)
    Country: str = Field(
        default="Other",
        description="Country of residence (automatically grouped to top countries or 'Other')",
        example="Canada",
    )
    Academic_Level: AcademicLevelEnum = Field(
        ..., description="Current educational stage", example=AcademicLevelEnum.undergraduate
    )
    Most_Used_Platform: PlatformEnum = Field(
        ..., description="Primary social media platform used", example=PlatformEnum.instagram
    )
    Purpose_Of_Use: PurposeOfUseEnum = Field(
        ..., description="Main purpose of using social media", example=PurposeOfUseEnum.entertainment
    )
    Avg_Daily_Usage_Hours: float = Field(
        ..., ge=0.0, le=24.0, description="Average daily social media screen time in hours", example=4.6
    )
    Daily_Unlocks: int = Field(
        ..., ge=0, le=1000, description="Average daily smartphone unlock count", example=166
    )
    Study_Hours: float = Field(
        ..., ge=0.0, le=24.0, description="Average daily academic study hours", example=4.0
    )
    Physical_Activity_Hours: float = Field(
        ..., ge=0.0, le=24.0, description="Average daily physical exercise / sports in hours", example=1.8
    )
    Sleep_Hours_Per_Night: float = Field(
        ..., ge=0.0, le=24.0, description="Average sleep duration per night in hours", example=6.7
    )
    Stress_Level: StressLevelEnum = Field(
        ..., description="Perceived academic/personal stress level", example=StressLevelEnum.medium
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "Age": 21,
                "Gender": "Male",
                "Country": "Canada",
                "Academic_Level": "Undergraduate",
                "Most_Used_Platform": "Instagram",
                "Purpose_Of_Use": "Entertainment",
                "Avg_Daily_Usage_Hours": 4.6,
                "Daily_Unlocks": 166,
                "Study_Hours": 4.0,
                "Physical_Activity_Hours": 1.8,
                "Sleep_Hours_Per_Night": 6.7,
                "Stress_Level": "Medium",
            }
        }
    }


class PredictionResult(BaseModel):
    predicted_mental_health_score: float = Field(
        ..., description="Predicted Mental Health Score (scale typically 1 to 10)", example=7.0
    )
    risk_category: str = Field(
        ..., description="Interpreted category based on predicted score", example="Good / Moderate"
    )
    status: str = Field(default="success", example="success")


class BatchPredictionRequest(BaseModel):
    students: List[StudentInput] = Field(..., description="List of student records to predict")


class BatchPredictionResponse(BaseModel):
    total: int = Field(..., example=1)
    predictions: List[PredictionResult]


# --- Helper Functions ---
def categorize_mental_health(score: float) -> str:
    """Categorizes the mental health score into risk tiers."""
    if score >= 7.5:
        return "Excellent / Low Risk"
    elif score >= 6.0:
        return "Good / Moderate"
    elif score >= 4.5:
        return "Fair / Mild Concern"
    else:
        return "Poor / High Risk"


def prepare_features(student: StudentInput) -> dict:
    """Formats student input into the exact feature DataFrame structure expected by the pipeline."""
    # Country grouping logic matching the trained model pipeline
    country_val = student.Country.strip()
    grouped_country = country_val if country_val in TOP_COUNTRIES else "Other"

    # Ensure non-negative physical activity matching EDA preprocessing
    physical_activity = max(0.0, float(student.Physical_Activity_Hours))

    return {
        "Study_Hours": float(student.Study_Hours),
        "Age": int(student.Age),
        "Avg_Daily_Usage_Hours": float(student.Avg_Daily_Usage_Hours),
        "Daily_Unlocks": int(student.Daily_Unlocks),
        "Physical_Activity_Hours": physical_activity,
        "Sleep_Hours_Per_Night": float(student.Sleep_Hours_Per_Night),
        "Stress_Level": student.Stress_Level.value,
        "Gender": student.Gender.value,
        "Academic_Level": student.Academic_Level.value,
        "Most_Used_Platform": student.Most_Used_Platform.value,
        "Purpose_Of_Use": student.Purpose_Of_Use.value,
        "Grouped_country": grouped_country,
    }


# --- API Routes ---
@app.get("/", tags=["General"])
def root():
    """Root endpoint providing service status and links to docs."""
    return {
        "message": "Student Social Media & Mental Health Impact Prediction API",
        "documentation": "/docs",
        "health": "/health",
        "model_loaded": model is not None,
    }


@app.get("/health", tags=["General"])
def health_check():
    """Health check endpoint to verify server and model status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_file_exists": os.path.exists(MODEL_FILE),
    }


@app.get("/model-info", tags=["Model"])
def model_info():
    """Returns metadata regarding features, required format, and model state."""
    return {
        "model_type": "RandomForestRegressor Pipeline with Skewed, Plain Numeric, Ordinal, and Nominal transformers",
        "target": "Mental_Health_Score",
        "top_countries_supported": sorted(list(TOP_COUNTRIES)),
        "stress_levels": [e.value for e in StressLevelEnum],
        "academic_levels": [e.value for e in AcademicLevelEnum],
        "platforms": [e.value for e in PlatformEnum],
        "purposes": [e.value for e in PurposeOfUseEnum],
    }


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
def predict_single(data: StudentInput):
    """Predict mental health score for an individual student."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Ensure Mental_Health_Model.pkl is present and valid.",
        )

    try:
        feature_dict = prepare_features(data)
        input_df = pd.DataFrame([feature_dict])
        prediction = float(model.predict(input_df)[0])
        prediction = round(prediction, 2)

        return PredictionResult(
            predicted_mental_health_score=prediction,
            risk_category=categorize_mental_health(prediction),
            status="success",
        )
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch_data: BatchPredictionRequest):
    """Predict mental health scores for multiple students at once."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Ensure Mental_Health_Model.pkl is present and valid.",
        )

    if not batch_data.students:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The students list cannot be empty.",
        )

    try:
        features_list = [prepare_features(s) for s in batch_data.students]
        input_df = pd.DataFrame(features_list)
        predictions = model.predict(input_df)

        results = []
        for pred in predictions:
            score = round(float(pred), 2)
            results.append(
                PredictionResult(
                    predicted_mental_health_score=score,
                    risk_category=categorize_mental_health(score),
                    status="success",
                )
            )

        return BatchPredictionResponse(total=len(results), predictions=results)
    except Exception as e:
        logger.error(f"Batch inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
