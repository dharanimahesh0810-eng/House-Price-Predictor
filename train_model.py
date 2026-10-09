"""
train_model.py
----------------
Trains the Bengaluru house price prediction model and saves it
so the Streamlit app (app.py) can load and use it.

Usage:
    python train_model.py

Requirements:
    Place "Bengaluru_House_Data.csv" in the same folder as this script.
"""

import pandas as pd
from category_encoders import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import joblib

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
print("Loading data...")
df = pd.read_csv("Bengaluru_House_Data.csv")

# ------------------------------------------------------------
# 2. Clean data
# ------------------------------------------------------------
print("Cleaning data...")

# Drop columns we don't need
df1 = df.drop(columns=['area_type', 'society', 'balcony', 'availability'])
df1.dropna(inplace=True)

# Extract number of bedrooms from "size" (e.g. "2 BHK" -> 2)
df1['bedrooms'] = df1['size'].apply(lambda x: int(x.split(" ")[0]))
df1 = df1.drop(columns='size')


def is_float(x):
    try:
        float(x)
    except ValueError:
        return False
    return True


def convert_sqft_to_num(x):
    if isinstance(x, float):
        return x
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except ValueError:
        return None


df4 = df1.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4.dropna(inplace=True)

# Remove sqft outliers (bottom/top 10%)
low, high = df4['total_sqft'].quantile([0.1, 0.9])
df4 = df4[df4['total_sqft'].between(low, high)]

# Remove unrealistic bathroom counts
df4 = df4[df4['bath'] < 12]

# Remove price-per-sqft outliers (bottom/top 10%)
df4['price_per_sqft'] = df4['price'] * 100000 / df4['total_sqft']
low, high = df4['price_per_sqft'].quantile([0.1, 0.9])
df4 = df4[df4['price_per_sqft'].between(low, high)]

# Group rare locations (<=10 occurrences) into "other"
df4.location = df4.location.apply(lambda x: x.strip())
location_stats = df4['location'].value_counts(ascending=False)
location_stats_less_than_10 = location_stats[location_stats <= 10]
df4.location = df4.location.apply(
    lambda x: 'other' if x in location_stats_less_than_10 else x
)

df4 = df4.drop(columns='price_per_sqft')

# ------------------------------------------------------------
# 3. Train / test split
# ------------------------------------------------------------
print("Splitting data...")
x = df4.drop(['price'], axis='columns')
y = df4.price

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=10
)

# ------------------------------------------------------------
# 4. Train model
# ------------------------------------------------------------
print("Training model...")
model = make_pipeline(OneHotEncoder(), LinearRegression())
model.fit(x_train, y_train)

# ------------------------------------------------------------
# 5. Evaluate model
# ------------------------------------------------------------
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f} Lakhs")
print(f"Mean Absolute Percentage Error: {mape * 100:.2f}%")

# ------------------------------------------------------------
# 6. Save model and location list for the Streamlit app
# ------------------------------------------------------------
print("Saving model...")
joblib.dump(model, 'house_price_model.pkl')

locations = sorted(x['location'].unique().tolist())
joblib.dump(locations, 'locations.pkl')

print("Done. Saved house_price_model.pkl and locations.pkl")
