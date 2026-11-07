import json
from pathlib import Path
import pandas as pd
import numpy as np

csv_path = Path("d:/Hotel_Revenue_Project/hotel_boooking/datasets/public_research/hotel_bookings_demand_antonio.csv")
df = pd.read_csv(csv_path)

print(f"=== STEP 2: DATA VALIDATION AUDIT ===")
print(f"File Path: {csv_path}")
print(f"Total Rows: {len(df):,}")
print(f"Total Columns: {len(df.columns)}")

print(f"\n1. Column Names & Dtypes:")
for col, dt in zip(df.columns, df.dtypes):
    print(f"  {col:32}: {str(dt):10} | Nulls: {df[col].isnull().sum():>6} ({df[col].isnull().mean()*100:.2f}%)")

print(f"\n2. Duplicate Records:")
full_dups = df.duplicated().sum()
print(f"  Full row duplicates: {full_dups:,} ({full_dups/len(df)*100:.2f}%)")

print(f"\n3. Target Column (is_canceled) Class Distribution:")
target_counts = df['is_canceled'].value_counts()
print(f"  Class 0 (Not Canceled / Stayed): {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/len(df)*100:.2f}%)")
print(f"  Class 1 (Canceled):              {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/len(df)*100:.2f}%)")

print(f"\n4. Arrival Date Range:")
# Reconstruct ISO date: year, month, day
# Antonio dataset has arrival_date_year, arrival_date_month (string), arrival_date_day_of_month
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
arrival_dates = pd.to_datetime(
    df['arrival_date_year'].astype(str) + '-' +
    df['arrival_date_month'].map(month_map).astype(str).str.zfill(2) + '-' +
    df['arrival_date_day_of_month'].astype(str).str.zfill(2),
    errors='coerce'
)
print(f"  Earliest Arrival Date: {arrival_dates.min().strftime('%Y-%m-%d')}")
print(f"  Latest Arrival Date:   {arrival_dates.max().strftime('%Y-%m-%d')}")
print(f"  Unique Arrival Dates:  {arrival_dates.nunique()} days across 26 months (July 2015 to August 2017)")

print(f"\n5. ADR Distribution:")
adr = df['adr']
print(f"  Mean:   ${adr.mean():.2f}")
print(f"  Std:    ${adr.std():.2f}")
print(f"  Median: ${adr.median():.2f}")
print(f"  Min:    ${adr.min():.2f}")
print(f"  25%:    ${adr.quantile(0.25):.2f}")
print(f"  75%:    ${adr.quantile(0.75):.2f}")
print(f"  Max:    ${adr.max():.2f}")
neg_adr = (adr < 0).sum()
print(f"  Negative ADR rows: {neg_adr} (e.g. refund/adjustment anomaly)")
zero_adr = (adr == 0).sum()
print(f"  Zero ADR rows:     {zero_adr} (complimentary/points stays)")

print(f"\n6. Booking Lead Time:")
lead = df['lead_time']
print(f"  Mean:   {lead.mean():.1f} days")
print(f"  Std:    {lead.std():.1f} days")
print(f"  Median: {lead.median():.1f} days")
print(f"  Min:    {lead.min()} days (same-day bookings)")
print(f"  Max:    {lead.max()} days")
print(f"  Cancellation rate for same-day bookings (lead_time=0): {df[df['lead_time']==0]['is_canceled'].mean()*100:.2f}%")
print(f"  Cancellation rate for lead_time > 100 days:            {df[df['lead_time']>100]['is_canceled'].mean()*100:.2f}%")

print(f"\n7. Customer Type Breakdown:")
for ct, grp in df.groupby('customer_type'):
    print(f"  {ct:20}: Count={len(grp):>6} ({len(grp)/len(df)*100:5.2f}%) | Cancel Rate={grp['is_canceled'].mean()*100:5.2f}%")

print(f"\n8. Booking Distribution Channel Breakdown:")
for ch, grp in df.groupby('distribution_channel'):
    print(f"  {ch:20}: Count={len(grp):>6} ({len(grp)/len(df)*100:5.2f}%) | Cancel Rate={grp['is_canceled'].mean()*100:5.2f}%")

print(f"\n9. Deposit Type Breakdown:")
for dt, grp in df.groupby('deposit_type'):
    print(f"  {dt:20}: Count={len(grp):>6} ({len(grp)/len(df)*100:5.2f}%) | Cancel Rate={grp['is_canceled'].mean()*100:5.2f}%")

