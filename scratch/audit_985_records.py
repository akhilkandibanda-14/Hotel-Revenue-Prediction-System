import os
from pathlib import Path
import pandas as pd
import numpy as np

csv_path = Path("d:/Hotel_Revenue_Project/web_app_dashboards/web_app/Hotel book.csv")

print(f"Target file: {csv_path}")
print(f"File exists: {csv_path.exists()}")
print(f"File size: {csv_path.stat().st_size} bytes")

df = pd.read_csv(csv_path)

print(f"\n--- 1. SHAPE & ROWS ---")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Columns list:\n{list(df.columns)}")

print(f"\n--- 2. COLUMN CHECK ---")
check_list = [
    "Booking ID", "booking_id", "Booking_ID", "ID",
    "Booking date", "booking_date",
    "Check-in date", "check_in_date",
    "Check-out date", "check_out_date",
    "Cancellation status", "is_canceled",
    "No-show status", "is_no_show",
    "ADR", "Booking_Channel", "Guest_Type", "Room_Type", "room_type",
    "Number of guests", "guests", "adults",
    "Reservation status", "reservation_status"
]
for item in check_list:
    found = [c for c in df.columns if c.lower() == item.lower() or c.lower() == item.replace(' ', '_').lower()]
    print(f"{item:25} -> {found if found else 'NOT FOUND'}")

print(f"\n--- 3. NUMERICAL SUMS & AGGREGATES ---")
agg_cols = ['Bookings', 'Cancellations', 'No_Shows', 'Checkins', 'Checkouts', 'New_Bookings', 'Reserved_Rooms', 'Available_Rooms']
for c in agg_cols:
    if c in df.columns:
        print(f"Sum of '{c}': {df[c].sum()} | Min: {df[c].min()} | Max: {df[c].max()} | Mean: {df[c].mean():.2f}")

print(f"\n--- 4. DUPLICATES ---")
print(f"Unique 'Date' entries: {df['Date'].nunique()} out of {len(df)}")
print(f"Duplicate rows across all columns: {df.duplicated().sum()}")

print(f"\n--- 5. CHANNEL BREAKDOWN ---")
if "Booking_Channel" in df.columns:
    ch_grp = df.groupby("Booking_Channel")[['Bookings', 'Cancellations', 'No_Shows']].sum()
    print(ch_grp)

print(f"\n--- 6. BOOKING RELATIONSHIPS ---")
print("Does Bookings == Checkins + Cancellations + No_Shows?")
print("Does Bookings == Reserved_Rooms?")
diff_reserved = (df['Bookings'] - df['Reserved_Rooms']).abs().sum()
print(f"Total absolute difference between 'Bookings' and 'Reserved_Rooms': {diff_reserved}")
sample_compare = df[['Date', 'Booking_Channel', 'Bookings', 'Reserved_Rooms', 'Cancellations', 'No_Shows', 'Checkins', 'Checkouts']].head(10)
print(sample_compare)

print("\n--- 7. DATE & MONTH PROGRESSION ---")
print("Month distribution:")
print(df['Month'].value_counts().sort_index())
for m in df['Month'].unique():
    sub = df[df['Month'] == m]
    print(f"Month {m}: {len(sub)} rows | start date={sub['Date'].iloc[0]} | end date={sub['Date'].iloc[-1]}")

print("\n--- 8. CANCELLATION & NO-SHOW OVERLAP CHECK ---")
print("Are Cancellations and No-Shows mutually exclusive or overlapping on the same day?")
print(pd.crosstab(df['Cancellations'], df['No_Shows'], margins=True))

print("\n--- 9. REVENUE IMPACT CALCULATION ---")
print("Can revenue associated with cancellations be determined from this dataset?")
print("Columns related to revenue:", [c for c in df.columns if 'rev' in c.lower()])
print("Average ADR:", df['ADR'].mean())
canc_with_adr = df[df['Cancellations'] > 0][['Date', 'Booking_Channel', 'Bookings', 'Cancellations', 'No_Shows', 'ADR', 'Total_Revenue']].head(10)
print(canc_with_adr)
