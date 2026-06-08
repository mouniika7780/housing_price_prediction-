from pydantic import BaseModel, Field
from typing import List
import datetime
class HouseFeatures(BaseModel):
    square_footage: float = Field(..., gt=0, description="Square footage of the house")
    bedrooms: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0.5, description="Number of bathrooms")
    year_built: int = Field(..., ge=1800, le=2025, description="Year the house was built")
    lot_size: float = Field(..., gt=0, description="Lot size in square feet")
    distance_to_city_center: float = Field(..., ge=0, description="Distance to city center in miles")
    school_rating: float = Field(..., ge=0, le=10, description="School district rating (0-10)")

    class Config:
        json_schema_extra = {
            "example": {
                "square_footage": 1550,
                "bedrooms": 3,
                "bathrooms": 2,
                "year_built": 1997,
                "lot_size": 6800,
                "distance_to_city_center": 4.1,
                "school_rating": 7.6
            }
        }

class BatchPredictionRequest(BaseModel):
    houses: List[HouseFeatures] = Field(..., min_items=1, max_items=100)

class PredictionResponse(BaseModel):
    predictions: List[float]
    count: int
    
class SinglePredictionResponse(BaseModel):
    prediction: float
    input_features: HouseFeatures

class ModelInfoResponse(BaseModel):
    model_type: str
    features: List[str]
    coefficients: dict
    metrics: dict
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str
    seconds: float
    service_time:datetime.timedelta