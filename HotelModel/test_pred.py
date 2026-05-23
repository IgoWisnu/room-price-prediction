import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

model = xgb.Booster()
model.load_model(r'c:\Python\rent-prediciton-fullapp\HotelModel\bali_price_model.json')

with open(r'c:\Python\rent-prediciton-fullapp\HotelModel\bali_preprocessor.pkl', 'rb') as f:
    preproc = pickle.load(f)

room = {
    "log_review":         np.log1p(190),
    "star_rating":        1,
    "rating":             4.3,
    "no_review":          0,
    "property_type_enc":  1,
    "room_size_m2":       34.0,
    "latitude":           -8.7,
    "longitude":          115.1,
    "is_luxury_brand":    0,
    "has_pool":           1,
    "has_wifi":           1,
    "has_ac":             1,
    "has_gym":            0,
    "has_spa":            0,
    "has_restaurant":     0,
    "has_parking":        0,
    "has_breakfast":      0,
    "has_kitchen":        0,
    "has_bathtub":        0,
    "has_balcony":        0,
    "has_ocean_view":     0,
    "is_free_cancellation": 0,
    "room_tier":          2,
    "value_added_count":  0,
    "has_pool_view":      0,
    "has_garden_view":    0,
    "has_ocean_view2":    0,
    "has_private_pool":   0,
    "has_pool_access":    0,
    "has_nature_view":    0,
    "has_city_view":      0,
    "kecamatan_cleaned":  "Kuta",
}

X = pd.DataFrame([room])
X["kecamatan_te"] = preproc["target_encoder"].transform(X[["kecamatan_cleaned"]])["kecamatan_cleaned"]
dmat = xgb.DMatrix(X[preproc["feature_cols"]])

raw_pred = model.predict(dmat)[0]
print("Raw pred:", raw_pred)
print("expm1:", np.expm1(raw_pred))
