import os
import pandas as pd

from app.utils import config


def save_reservation(data: dict):

    os.makedirs(config.RESERVATION_FOLDER, exist_ok=True)

    data = data.copy()
    data["Predicted_Revenue"] = None

    df = pd.DataFrame([data])

    if os.path.exists(config.RESERVATION_PATH):
        df.to_csv(config.RESERVATION_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(config.RESERVATION_PATH, index=False)

    return None, None
