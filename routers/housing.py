from fastapi import  APIRouter,HTTPException
from schemas import (HouseFeatures, BatchPredictionRequest,PredictionResponse, SinglePredictionResponse,ModelInfoResponse, HealthResponse)
from models import HousingModel
import time
import datetime

router = APIRouter()

housing_instance = HousingModel.load("saved_models/housing_price_model.pkl")
start_time = datetime.datetime.now()
print("Housing model loaded with metrics:", housing_instance if housing_instance else "No model found")

@router.get("/health", response_model=HealthResponse)
async def health_check():
    end_time = datetime.datetime.now()
    service_time =  end_time - start_time 


    return HealthResponse(
        status="healthy",
        service_time=service_time,
        seconds= service_time.total_seconds(),
        model_loaded= housing_instance is not None and housing_instance.is_trained,
        message="API is running" if housing_instance else "API running but model not loaded"
        )

@router.post("/predict", response_model=SinglePredictionResponse)
async def predict_single(house: HouseFeatures):
    # import pdb; pdb.set_trace()
    # print("Single prediction request received with features:", housing_instance)
    if not housing_instance or not housing_instance.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    features = [house.model_dump()]
    prediction = housing_instance.predict(features)[0]
    return SinglePredictionResponse(prediction=prediction,input_features=house)


@router.post("/predict/batch", response_model=PredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    if not housing_instance or not housing_instance.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    features = [h.model_dump() for h in request.houses]
    predictions = housing_instance.predict(features)
    return PredictionResponse(predictions=predictions,count=len(predictions))



@router.post("/predict/all", response_model=PredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    if not housing_instance or not housing_instance.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    features = [h.model_dump() for h in request.houses]
    # import pdb; pdb.set_trace()

    if len(features) == 1:
        predictions = housing_instance.predict(features)
        print("Single prediction made:", predictions)
    else:
        predictions = housing_instance.predict(features)
        print("Batch prediction made:", predictions)

    

    # features = [h.model_dump() for h in request.houses]
    # predictions = housing_instance.predict(features)

    
    return PredictionResponse(predictions=predictions,count=len(predictions))



@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    if not housing_instance or not housing_instance.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return ModelInfoResponse(model_type="Linear Regression",features=housing_instance.feature_names,coefficients=housing_instance.get_coefficients(),metrics=housing_instance.metrics)