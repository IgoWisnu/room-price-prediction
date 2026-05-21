import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="Bali Room Price API", version="4.2")

# --- 1. LOAD MODEL & PREPROCESSOR ---
model = xgb.Booster()
model.load_model("bali_price_model.json")

with open("bali_preprocessor.pkl", "rb") as f:
    preproc = pickle.load(f)

# --- 2. DATA METADATA UNTUK FRONTEND ---
METADATA = {
    "room_tiers": [
        {"id": 1, "name": "Standard / Budget / Classic"},
        {"id": 2, "name": "Superior / Deluxe / Comfort"},
        {"id": 3, "name": "Premier / Junior Suite / Executive"},
        {"id": 4, "name": "Suite / Family Suite"},
        {"id": 5, "name": "Villa / Presidential / Penthouse"}
    ],
    "property_types": [
        {"id": 0, "name": "Capsule Hotel"},
        {"id": 1, "name": "Guest House"},
        {"id": 2, "name": "Homestay"},
        {"id": 3, "name": "Hostel"},
        {"id": 4, "name": "Hotel"},
        {"id": 5, "name": "Private Aparthotel"},
        {"id": 6, "name": "Private Hostel"},
        {"id": 7, "name": "Resort"},
        {"id": 8, "name": "Resort Villa"}
    ],
    "exclusive_features": [
        "None", "Private Pool", "Ocean View", "Pool View", 
        "Garden View", "Pool Access", "Nature View", "City View"
    ],
    "kecamatan": [
        "Kuta", "Seminyak", "Legian", "Canggu", "Jimbaran", 
        "Nusa Dua", "Ubud", "Sanur", "Uluwatu", "Tegallalang", 
        "Lovina", "Kintamani", "Candidasa", "West Denpasar", "South Denpasar"
    ],
    "standard_facilities": [
        "AC", "WiFi", "Pool", "Gym", "Spa", "Restaurant", 
        "Parking", "Breakfast", "Kitchen", "Bathtub", "Free Cancellation"
    ]
}

# --- 3. SCHEMA INPUT ---
class PredictionInput(BaseModel):
    hotel_name: str
    kecamatan: str
    star_rating: int
    rating: float
    review_count: int
    # Lat Long sekarang bisa dikirim dari FE
    latitude: float
    longitude: float
    room_size_m2: Optional[float] = None
    is_luxury: int         # 0 atau 1
    room_tier: int         # 1-5
    property_type_enc: int # 0-8
    exclusive_feature: str # Pilih dari METADATA["exclusive_features"]
    facilities: List[str]  # Pilih dari METADATA["standard_facilities"]

# --- 4. ENDPOINT METADATA (GET) ---
@app.get("/metadata")
async def get_metadata():
    """Endpoint untuk membantu Frontend mengisi pilihan Dropdown/Radio Button"""
    return METADATA

# --- 5. ENDPOINT PREDIKSI (POST) ---
@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        # Mapping exclusive feature
        feat = data.exclusive_feature.lower()
        mapping = {
            "has_pool_view": 1 if "pool view" in feat else 0,
            "has_garden_view": 1 if "garden" in feat else 0,
            "has_ocean_view2": 1 if any(x in feat for x in ['ocean', 'sea', 'beach']) else 0,
            "has_private_pool": 1 if "private pool" in feat else 0,
            "has_pool_access": 1 if "pool access" in feat else 0,
            "has_nature_view": 1 if any(x in feat for x in ['jungle', 'valley', 'nature']) else 0,
            "has_city_view": 1 if "city" in feat else 0,
        }

        # Susun input data sesuai urutan model
        input_dict = {
            "log_review": np.log1p(data.review_count),
            "star_rating": data.star_rating,
            "rating": data.rating,
            "no_review": 1 if data.rating == 0 else 0,
            "room_size_m2": data.room_size_m2 if data.room_size_m2 else preproc["room_size_m2_median"],
            "latitude": data.latitude,
            "longitude": data.longitude,
            "is_luxury_brand": data.is_luxury,
            "room_tier": data.room_tier,
            "property_type_enc": data.property_type_enc,
            "has_pool": 1 if "Pool" in data.facilities else 0,
            "has_wifi": 1 if "WiFi" in data.facilities else 0,
            "has_ac": 1 if "AC" in data.facilities else 0,
            "has_gym": 1 if "Gym" in data.facilities else 0,
            "has_spa": 1 if "Spa" in data.facilities else 0,
            "has_restaurant": 1 if "Restaurant" in data.facilities else 0,
            "has_parking": 1 if "Parking" in data.facilities else 0,
            "has_breakfast": 1 if "Breakfast" in data.facilities else 0,
            "has_kitchen": 1 if "Kitchen" in data.facilities else 0,
            "has_bathtub": 1 if "Bathtub" in data.facilities else 0,
            "has_balcony": 0,
            "has_ocean_view": 0,
            "is_free_cancellation": 1 if "Free Cancellation" in data.facilities else 0,
            "value_added_count": 1 if len(data.facilities) > 5 else 0,
            **mapping, # Masukkan hasil mapping view tadi
            "kecamatan_cleaned": data.kecamatan
        }

        X = pd.DataFrame([input_dict])
        
        # Target Encoding Kecamatan
        X["kecamatan_te"] = preproc["target_encoder"].transform(X[["kecamatan_cleaned"]])["kecamatan_cleaned"]
        
        # Inferensi XGBoost
        dmat = xgb.DMatrix(X[preproc["feature_cols"]])
        log_p = model.predict(dmat)[0]
        price_idr = int(np.expm1(log_p))

        # Rentang Wajar (25% error margin)
        margin = int(price_idr * 0.25)
        
        return {
            "status": "success",
            "prediction": {
                "estimated_price": price_idr,
                "range_min": price_idr - margin,
                "range_max": price_idr + margin,
                "currency": "IDR"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)