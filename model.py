import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# dataset load
df = pd.read_csv("govdata.csv")

# unnecessary columns remove
df = df.drop(columns=["S. No.", "State/ UT"], errors="ignore")
df = df.select_dtypes(include=[float,int])

# ── Build training rows: one row per (state × weather_condition) ─────────────
# Features  → 2014 data (accidents, killed, injured, weather code, mortality)
# Target    → 2016 accidents for same weather category

weather_map = {
    "Fine/Clear":  ("Fine - Total Acc. - 2014",
                    "Fine - Persons Killed - 2014",
                    "Fine - Persons Injured - 2014",
                    "Fine/Clear - Total Accidents - 2016"),
    "Mist/Fog":    ("Mist/fog - Total Acc. - 2014",
                    "Mist/fog - Persons Killed - 2014",
                    "Mist/fog - Persons Injured - 2014",
                    "Mist/ Foggy - Total Accidents - 2016"),
    "Cloudy":      ("Cloudy - Total Acc. - 2014",
                    "Cloudy - Persons Killed - 2014",
                    "Cloudy - Persons Injured - 2014",
                    "Cloudy - Total Accidents - 2016"),
    "Light Rain":  ("Light rain - Total Acc. - 2014",
                    "Light rain - Persons Killed - 2014",
                    "Light rain - Persons Injured - 2014",
                    "Rainy - Total Accidents - 2016"),
    "Heavy Rain":  ("Heavy rain - Total Acc. - 2014",
                    "Heavy rain - Persons Killed - 2014",
                    "Heavy rain - Persons Injured - 2014",
                    "Rainy - Total Accidents - 2016"),
    "Flooding":    ("Flooding of slipways/rivulers - Total Acc. - 2014",
                    "Flooding of slipways/rivulers - Persons Killed - 2014",
                    "Flooding of slipways/rivulers - Persons Injured - 2014",
                    "Rainy - Total Accidents - 2016"),
    "Hail/Sleet":  ("Hail/sleet - Total Acc. - 2014",
                    "Hail/sleet - Persons Killed - 2014",
                    "Hail/sleet - Persons Injured - 2014",
                    "Hail/Sleet - Total Accidents - 2016"),
    "Snow":        ("snow - Total Acc. - 2014",
                    "snow - Persons Killed - 2014",
                    "snow - Persons Injured - 2014",
                    "Snowfall - Total Accidents - 2016"),
    "Strong Wind": ("Strong wind - Total Acc. - 2014",
                    "Strong wind - Persons Killed - 2014",
                    "Strong wind - Persons Injured - 2014",
                    "Others - Total Accidents - 2016"),
    "Dust Storm":  ("Dust storm - Total Acc. - 2014",
                    "Dust storm - Persons Killed - 2014",
                    "Dust storm - Persons Injured - 2014",
                    "Others - Total Accidents - 2016"),
    "Very Hot":    ("Very hot - Total Acc. - 2014",
                    "Very hot - Persons Killed - 2014",
                    "Very hot - Persons Injured - 2014",
                    "Others - Total Accidents - 2016"),
    "Very Cold":   ("Very cold - Total Acc. - 2014",
                    "Very cold - Persons Killed - 2014",
                    "Very cold - Persons Injured - 2014",
                    "Others - Total Accidents - 2016"),
    "Others":      ("Other extraordinary weather condition - Total Acc. - 2014",
                    "Other extraordinary weather condition - Persons Killed - 2014",
                    "Other extraordinary weather condition - Persons Injured - 2014",
                    "Others - Total Accidents - 2016"),
}

rows = []
for i, row in df.iterrows():
    for w_idx, (weather, (acc14, kill14, inj14, acc16)) in enumerate(weather_map.items()):
        acc_val = row[acc14] if row[acc14] > 0 else 0
        rows.append({
            "acc_2014":       acc_val,
            "killed_2014":    row[kill14],
            "injured_2014":   row[inj14],
            "weather_code":   w_idx,
            "mortality_rate": (row[kill14] / acc_val) if acc_val > 0 else 0,
            "target_acc_2016": row[acc16],
        })

train_df = pd.DataFrame(rows)

X = train_df[["acc_2014", "killed_2014", "injured_2014", "weather_code", "mortality_rate"]]
y = train_df["target_acc_2016"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model create
model = RandomForestRegressor(n_estimators=100, random_state=42)

# train
model.fit(X_train, y_train)
print("Model Training Completed")

# prediction
pred = model.predict(X_test)

# accuracy
score = r2_score(y_test, pred)
print("Model Accuracy (R2 Score):", round(score, 4))

# save model
joblib.dump(model, "accident_model.pkl")
print("Model saved successfully")
