# Hotel Revenue Prediction & Dashboard

Interactive Streamlit app for uploading a hotel booking dataset, training a revenue prediction model, and visualizing the dashboards from your original script.

## What it shows

- Revenue vs profit scatter plot
- Monthly revenue trend
- Occupancy rate by season
- Booking channel distribution
- Top guest countries
- Actual vs predicted revenue
- Feature importance
- Dataset preview and CSV downloads

## Run locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
streamlit run app.py
```

## Data

Upload your CSV in the sidebar. The app also checks for a file named `Hotel book.csv` in the project folder.

## Notes

- The model uses `Total_Revenue` as the default target if it exists.
- If your dataset has different column names, the app will still load and show the available charts that match the columns present.