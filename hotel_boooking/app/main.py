from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.prediction import router as prediction_router
from app.api.upload import router as upload_router
from app.api.reservation import router as reservation_router
from app.scheduler.scheduler import training_scheduler
from app.utils.logger import setup_logging

# Initialize standard logging configuration
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    training_scheduler.start()
    yield
    training_scheduler.stop()


app = FastAPI(
    title="Hotel Revenue Prediction Service",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(prediction_router)
app.include_router(upload_router)
app.include_router(reservation_router)
app.include_router(health_router)


@app.get("/")
def home():
    return {
        "message": "Hotel Revenue Service Running"
    }