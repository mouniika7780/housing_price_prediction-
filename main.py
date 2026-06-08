from fastapi import FastAPI
from contextlib import asynccontextmanager
import os
from routers import housing_router
from models import HousingModel
import datetime

model: HousingModel = None
global start_time

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    start_time = datetime.datetime.now()
    model_path = "saved_models/housing_price_model.pkl"
    if os.path.exists(model_path):
        model = HousingModel.load(model_path)

    yield
    model = None

app = FastAPI(
    title="Housing Price Prediction API",
    description="Predict housing prices",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(housing_router, prefix="/api/v1", tags=["Housing Price Prediction"])


