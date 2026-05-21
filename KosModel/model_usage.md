# XGBoost Bali Kos Price Predictor - Model Usage Docs

This document outlines how to use the pre-trained XGBoost model for predicting monthly kos (room) prices in Bali.

## 1. Requirements

Before using the model, ensure you have the following installed:
```bash
pip install xgboost pandas numpy scikit-learn
```

## 2. Model File

The trained model is saved in JSON format, which can be loaded directly using XGBoost:
- **`xgboost_bali_kos_price.json`**

## 3. Loading the Model

```python
import xgboost as xgb

# Initialize a new model
model = xgb.XGBRegressor()

# Load the saved model
model.load_model("xgboost_bali_kos_price.json")
```

## 4. Input Features

The model expects exactly the same features (in the same order) as seen during training. You should pass a `pandas.DataFrame` or a 2D `numpy.array` with the following features:

1. **Amenity Columns (1 or 0)**:
   - `ac`, `air_panas`, `kamar_mandi_dalam`, `parkir`, `dispenser`, `kulkas`, `kursi`, `lemari`, `meja`, `parkir_mobil`, `ruang_tamu`, `shower`, `tv`, `wastafel`, `wifi`, `cctv`, `swimming_pool`, `kasur_bed_springbed`
   
2. **Engineered Features**:
   - `amenity_count`: Total number of the above amenities.
   - `premium_count`: Count of premium amenities (`ac`, `air_panas`, `kamar_mandi_dalam`, `kulkas`, `swimming_pool`, `tv`, `wifi`, `kasur_bed_springbed`).
   - `amenity_density`: `amenity_count` divided by the total number of possible amenities (18).
   - `has_private_bath_ac`: 1 if both `kamar_mandi_dalam` == 1 and `ac` == 1, else 0.
   - `luxury_score`: Count of luxury amenities (`ac`, `air_panas`, `kulkas`, `tv`, `wifi`, `swimming_pool`).
   - `is_luxury`: 1 if `luxury_score` >= 4, else 0.
   - `location_tier`: 2 for Tourist Areas (badung, jimbaran, abiansemal), 1 for City Areas (denpasar, mengwi, kediri), 0 for Budget Areas.
   - `location_enc`: A numerical encoding of the specific region (based on the LabelEncoder mapping created during training). For a robust approach if you don't have the LabelEncoder object saved, you may recreate it dynamically or stick to 0 for unknown.

## 5. Inference Example

```python
import pandas as pd
import xgboost as xgb

model = xgb.XGBRegressor()
model.load_model("xgboost_bali_kos_price.json")

# Example: Mid-range kos in Badung
input_data = {
    "ac": 1, "air_panas": 1, "kamar_mandi_dalam": 1, "parkir": 1,
    "dispenser": 1, "kulkas": 0, "kursi": 1, "lemari": 1, "meja": 1,
    "parkir_mobil": 0, "ruang_tamu": 0, "shower": 1, "tv": 1,
    "wastafel": 1, "wifi": 1, "cctv": 0, "swimming_pool": 0,
    "kasur_bed_springbed": 1,
    "amenity_count": 13,
    "premium_count": 7,
    "amenity_density": 13 / 18,
    "has_private_bath_ac": 1,
    "luxury_score": 4,
    "is_luxury": 1,
    "location_tier": 2, # Tourist Area
    "location_enc": 1   # Badung (based on internal encoder)
}

# Create a DataFrame
df_input = pd.DataFrame([input_data])

# Make prediction
predicted_price = model.predict(df_input)[0]

print(f"Predicted Price: Rp {predicted_price:,.0f} / month")
```

## 6. Output interpretation
The output will be a single float representing the predicted monthly rental price in Indonesian Rupiah (IDR).
