import os
import pandas as pd

from app.utils.config import RESERVATION_FOLDER, RESERVATION_PATH


def save_reservation(data: dict):

    os.makedirs(RESERVATION_FOLDER, exist_ok=True)

    data = data.copy()
    data["Predicted_Revenue"] = None

    df = pd.DataFrame([data])

    if os.path.exists(RESERVATION_PATH):
        df.to_csv(RESERVATION_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(RESERVATION_PATH, index=False)

    return None, None
