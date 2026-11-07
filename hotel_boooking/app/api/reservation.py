from fastapi import APIRouter
from pydantic import BaseModel

from app.services.reservation_service import save_reservation

router = APIRouter(
    prefix="/reservation",
    tags=["Reservation"]
)


class ReservationRequest(BaseModel):
    Date: str
    Month: str
    Booking_Channel: str
    Guest_Type: str
    Market_Segment: str
    Guest_Country: str
    Season: str

    Occupancy_Rate: float
    ADR: float
    RevPAR: float

    Available_Rooms: int
    Reserved_Rooms: int

    Fixed_Costs: float
    Variable_Costs: float
    Total_Costs: float


@router.post("/")
def reservation(request: ReservationRequest):

    try:

        save_reservation(
            request.model_dump()
        )

        return {
            "status": "Success",
            "message": "Reservation logged successfully"
        }

    except Exception as exc:

        return {
            "status": "Failed",
            "message": str(exc)
        }
