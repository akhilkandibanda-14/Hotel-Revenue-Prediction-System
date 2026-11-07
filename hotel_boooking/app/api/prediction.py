from fastapi import APIRouter
from pydantic import BaseModel

from app.services.predictor import RevenuePredictor

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)

predictor = RevenuePredictor()


class PredictionRequest(BaseModel):
    Date: str


@router.post("/")
def predict(request: PredictionRequest):

    try:

        revenue = predictor.predict({"Date": request.Date})

        return {

            "status": "Success",

            "Date": request.Date,

            "Predicted_Revenue": revenue

        }

    except Exception as e:

        return {

            "status": "Failed",

            "message": str(e)

        }