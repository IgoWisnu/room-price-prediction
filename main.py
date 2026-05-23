"""
Unified Bali Rent Price Prediction API
=====================================
Endpoints:
  POST /predict/pernight   — Hotel/Villa nightly price (IDR)
  POST /predict/permonth   — Kos monthly price (IDR)
  GET  /metadata/pernight  — Dropdown options for hotel form
  GET  /metadata/permonth  — Dropdown options for kos form
  GET  /health             — Health check

Both /predict endpoints return:
  - prediction (estimated price + range)
  - similar_properties (top-5 nearest from training data)

Run:
  pip install fastapi uvicorn xgboost category_encoders scikit-learn pandas numpy
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import pickle
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
HOTEL_DIR  = BASE_DIR / "HotelModel"
KOS_DIR    = BASE_DIR / "KosModel"

# ---------------------------------------------------------------------------
# Load Hotel Model
# ---------------------------------------------------------------------------
logger.info("Loading Hotel (per-night) model …")
hotel_model = xgb.Booster()
hotel_model.load_model(str(HOTEL_DIR / "bali_price_model.json"))

with open(HOTEL_DIR / "bali_preprocessor.pkl", "rb") as _f:
    hotel_preproc = pickle.load(_f)

hotel_df = pd.read_csv(HOTEL_DIR / "bali_cleaned.csv")
logger.info(f"  Hotel dataset loaded: {len(hotel_df):,} rows")

# ---------------------------------------------------------------------------
# Load Kos Model
# ---------------------------------------------------------------------------
logger.info("Loading Kos (per-month) model …")
kos_model = xgb.XGBRegressor()

# Support both pkl (from training pipeline) and json (from docs)
_kos_pkl  = KOS_DIR / "xgboost_bali_kos_price.pkl"
_kos_json = KOS_DIR / "xgboost_bali_kos_price.json"  # fallback (docs mention json)
_kos_json2 = KOS_DIR / "bali_kos_price.json"

if _kos_pkl.exists():
    with open(_kos_pkl, "rb") as _f:
        kos_model = pickle.load(_f)
    logger.info("  Kos model loaded from .pkl")
elif _kos_json.exists():
    kos_model.load_model(str(_kos_json))
    logger.info("  Kos model loaded from xgboost_bali_kos_price.json")
elif _kos_json2.exists():
    kos_model.load_model(str(_kos_json2))
    logger.info("  Kos model loaded from bali_kos_price.json")
else:
    raise FileNotFoundError("Kos model file not found in KosModel/")

kos_df = pd.read_csv(KOS_DIR / "kos_data_cleaned.csv")
logger.info(f"  Kos dataset loaded: {len(kos_df):,} rows")

# ---------------------------------------------------------------------------
# Kos Location Maps (derived from training data)
# ---------------------------------------------------------------------------
KOS_LOCATION_MAP: Dict[str, int] = (
    kos_df[["location", "location_enc"]]
    .drop_duplicates()
    .set_index("location")["location_enc"]
    .to_dict()
)

KOS_TIER_MAP: Dict[str, int] = {
    "abiansemal": 2, "jimbaran": 2, "badung": 2,        # Tourist
    "denpasar": 1, "mengwi": 1, "kediri": 1,             # City
    "bali": 0, "buleleng": 0, "gianyar": 0,
    "karangasem": 0, "tabanan": 0,                        # Budget / Other
}

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bali Rent Price Prediction API",
    description=(
        "Predict nightly room prices (hotel/villa) or monthly kos (room rental) "
        "prices in Bali, Indonesia. Both endpoints also return similar properties "
        "from the training dataset."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================================
# METADATA — HOTEL
# ===========================================================================
HOTEL_METADATA = {
    "room_tiers": [
        {"id": 1, "name": "Standard / Budget / Classic"},
        {"id": 2, "name": "Superior / Deluxe / Comfort"},
        {"id": 3, "name": "Premier / Junior Suite / Executive"},
        {"id": 4, "name": "Suite / Family Suite"},
        {"id": 5, "name": "Villa / Presidential / Penthouse"},
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
        {"id": 8, "name": "Resort Villa"},
    ],
    "exclusive_features": [
        "None", "Private Pool", "Ocean View", "Pool View",
        "Garden View", "Pool Access", "Nature View", "City View",
    ],
    "kecamatan": [
        "Kuta", "Seminyak", "Legian", "Canggu", "Jimbaran",
        "Nusa Dua", "Ubud", "Sanur", "Uluwatu", "Tegallalang",
        "Lovina", "Kintamani", "Candidasa", "West Denpasar", "South Denpasar",
    ],
    "standard_facilities": [
        "AC", "WiFi", "Pool", "Gym", "Spa", "Restaurant",
        "Parking", "Breakfast", "Kitchen", "Bathtub", "Free Cancellation",
    ],
    "luxury_brands_note": (
        "Mark is_luxury_brand=1 if the hotel belongs to a chain such as: "
        "Aman, Bulgari, Four Seasons, Ritz-Carlton, St. Regis, Six Senses, "
        "Raffles, Alila, Anantara, W Bali, Ayana, Mulia, Sofitel, Hilton, "
        "Hyatt, Sheraton, etc."
    ),
}

# ===========================================================================
# METADATA — KOS
# ===========================================================================
KOS_METADATA = {
    "amenities": [
        {"id": "ac",                  "label": "AC (Air Conditioner)"},
        {"id": "air_panas",           "label": "Water Heater"},
        {"id": "kamar_mandi_dalam",   "label": "Private Bathroom"},
        {"id": "parkir",              "label": "Motorcycle Parking"},
        {"id": "dispenser",           "label": "Water Dispenser"},
        {"id": "kulkas",              "label": "Refrigerator"},
        {"id": "kursi",               "label": "Chair"},
        {"id": "lemari",              "label": "Wardrobe"},
        {"id": "meja",                "label": "Desk/Table"},
        {"id": "parkir_mobil",        "label": "Car Parking"},
        {"id": "ruang_tamu",          "label": "Living Room"},
        {"id": "shower",              "label": "Shower"},
        {"id": "tv",                  "label": "TV"},
        {"id": "wastafel",            "label": "Washbasin"},
        {"id": "wifi",                "label": "WiFi"},
        {"id": "cctv",                "label": "CCTV"},
        {"id": "swimming_pool",       "label": "Swimming Pool"},
        {"id": "kasur_bed_springbed", "label": "Spring Bed"},
    ],
    "locations": [
        {"id": "abiansemal", "enc": 0,  "tier": 2, "tier_name": "Tourist Area"},
        {"id": "badung",     "enc": 1,  "tier": 2, "tier_name": "Tourist Area"},
        {"id": "bali",       "enc": 2,  "tier": 0, "tier_name": "Budget Area"},
        {"id": "buleleng",   "enc": 3,  "tier": 0, "tier_name": "Budget Area"},
        {"id": "denpasar",   "enc": 4,  "tier": 1, "tier_name": "City Area"},
        {"id": "gianyar",    "enc": 5,  "tier": 0, "tier_name": "Budget Area"},
        {"id": "jimbaran",   "enc": 6,  "tier": 2, "tier_name": "Tourist Area"},
        {"id": "karangasem", "enc": 7,  "tier": 0, "tier_name": "Budget Area"},
        {"id": "kediri",     "enc": 8,  "tier": 1, "tier_name": "City Area"},
        {"id": "mengwi",     "enc": 9,  "tier": 1, "tier_name": "City Area"},
        {"id": "tabanan",    "enc": 10, "tier": 0, "tier_name": "Budget Area"},
    ],
    "premium_amenities": ["ac", "air_panas", "kamar_mandi_dalam", "kulkas",
                          "swimming_pool", "tv", "wifi", "kasur_bed_springbed"],
    "luxury_amenities":  ["ac", "air_panas", "kulkas", "tv", "wifi", "swimming_pool"],
}

# ===========================================================================
# SCHEMAS — HOTEL
# ===========================================================================
class HotelPredictInput(BaseModel):
    # Identity / context (not used in model, only for display)
    hotel_name: Optional[str] = Field(None, description="Hotel/villa name for display only")

    # Location
    kecamatan: str = Field(..., description="Area in Bali — see /metadata/pernight for valid values")
    latitude:  float = Field(..., ge=-9.0, le=-8.0)
    longitude: float = Field(..., ge=114.5, le=116.0)

    # Hotel level
    star_rating:      int   = Field(..., ge=0, le=5)
    rating:           float = Field(..., ge=0.0, le=10.0, description="Guest score 0–10; use 0 if no reviews")
    review_count:     int   = Field(..., ge=0)
    is_luxury_brand:  int   = Field(0, ge=0, le=1)
    is_luxury:        Optional[int] = Field(None, description="Alias for is_luxury_brand")
    property_type_enc: int  = Field(4, ge=0, le=8, description="See /metadata/pernight property_types")

    # Convenience frontend lists
    facilities:       Optional[List[str]] = Field(None, description="Array of strings like 'AC', 'Gym'")
    exclusive_feature: Optional[str] = Field(None, description="E.g., 'Private Pool', 'Ocean View'")

    # Room details
    room_tier:   int           = Field(..., ge=1, le=5)
    room_size_m2: Optional[float] = Field(None, ge=1)

    # Facilities (hotel-level)
    has_pool:       int = Field(0, ge=0, le=1)
    has_wifi:       int = Field(1, ge=0, le=1)
    has_ac:         int = Field(1, ge=0, le=1)
    has_gym:        int = Field(0, ge=0, le=1)
    has_spa:        int = Field(0, ge=0, le=1)
    has_restaurant: int = Field(0, ge=0, le=1)
    has_parking:    int = Field(0, ge=0, le=1)

    # Package / room features
    has_breakfast:        int = Field(0, ge=0, le=1)
    has_kitchen:          int = Field(0, ge=0, le=1)
    has_bathtub:          int = Field(0, ge=0, le=1)
    is_free_cancellation: int = Field(0, ge=0, le=1)
    value_added_count:    int = Field(0, ge=0)

    # Exclusive view / pool feature (room-name based)
    has_pool_view:    int = Field(0, ge=0, le=1)
    has_garden_view:  int = Field(0, ge=0, le=1)
    has_ocean_view2:  int = Field(0, ge=0, le=1, description="Ocean/sea/beach view in room name")
    has_private_pool: int = Field(0, ge=0, le=1)
    has_pool_access:  int = Field(0, ge=0, le=1)
    has_nature_view:  int = Field(0, ge=0, le=1)
    has_city_view:    int = Field(0, ge=0, le=1)

    # Similarity search options
    similar_count: int = Field(5, ge=1, le=20, description="Number of similar properties to return")


# ===========================================================================
# SCHEMAS — KOS
# ===========================================================================
class KosPredictInput(BaseModel):
    # Identity / context (not used in model, only for display)
    kos_name: Optional[str] = Field(None, description="Kos name for display only")

    # Amenities (0/1)
    ac:                  int = Field(0, ge=0, le=1)
    air_panas:           int = Field(0, ge=0, le=1)
    kamar_mandi_dalam:   int = Field(0, ge=0, le=1)
    parkir:              int = Field(0, ge=0, le=1)
    dispenser:           int = Field(0, ge=0, le=1)
    kulkas:              int = Field(0, ge=0, le=1)
    kursi:               int = Field(0, ge=0, le=1)
    lemari:              int = Field(0, ge=0, le=1)
    meja:                int = Field(0, ge=0, le=1)
    parkir_mobil:        int = Field(0, ge=0, le=1)
    ruang_tamu:          int = Field(0, ge=0, le=1)
    shower:              int = Field(0, ge=0, le=1)
    tv:                  int = Field(0, ge=0, le=1)
    wastafel:            int = Field(0, ge=0, le=1)
    wifi:                int = Field(0, ge=0, le=1)
    cctv:                int = Field(0, ge=0, le=1)
    swimming_pool:       int = Field(0, ge=0, le=1)
    kasur_bed_springbed: int = Field(0, ge=0, le=1)

    # Location
    location: str = Field(..., description="Location in Bali — see /metadata/permonth for valid values")

    # Similarity search options
    similar_count: int = Field(5, ge=1, le=20)


# ===========================================================================
# HELPER — Compute engineered Kos features
# ===========================================================================
PREMIUM_AMENITIES = {"ac", "air_panas", "kamar_mandi_dalam", "kulkas",
                     "swimming_pool", "tv", "wifi", "kasur_bed_springbed"}
LUXURY_AMENITIES  = {"ac", "air_panas", "kulkas", "tv", "wifi", "swimming_pool"}
ALL_AMENITIES     = [
    "ac", "air_panas", "kamar_mandi_dalam", "parkir", "dispenser", "kulkas",
    "kursi", "lemari", "meja", "parkir_mobil", "ruang_tamu", "shower", "tv",
    "wastafel", "wifi", "cctv", "swimming_pool", "kasur_bed_springbed",
]

def compute_kos_features(d: KosPredictInput) -> dict:
    raw = d.model_dump()
    amenity_vals = {k: raw[k] for k in ALL_AMENITIES}

    amenity_count   = sum(amenity_vals.values())
    premium_count   = sum(amenity_vals[k] for k in PREMIUM_AMENITIES)
    luxury_score    = sum(amenity_vals[k] for k in LUXURY_AMENITIES)
    amenity_density = amenity_count / len(ALL_AMENITIES)

    loc = d.location.lower().strip()
    location_enc  = KOS_LOCATION_MAP.get(loc, 2)   # default "bali"
    location_tier = KOS_TIER_MAP.get(loc, 0)

    return {
        **amenity_vals,
        "amenity_count":       amenity_count,
        "premium_count":       premium_count,
        "amenity_density":     amenity_density,
        "has_private_bath_ac": 1 if (amenity_vals["kamar_mandi_dalam"] and amenity_vals["ac"]) else 0,
        "luxury_score":        luxury_score,
        "is_luxury":           1 if luxury_score >= 4 else 0,
        "location_tier":       location_tier,
        "location_enc":        location_enc,
    }


# ===========================================================================
# HELPER — Similar Properties for Hotel
# ===========================================================================
def get_similar_hotels(input_dict: dict, kecamatan: str, n: int = 5) -> List[Dict]:
    """
    Find similar hotels from the training CSV using facility-based cosine similarity.
    Prefers properties in the same kecamatan; falls back to whole dataset.
    """
    flag_cols = [
        "has_pool", "has_wifi", "has_ac", "has_gym", "has_spa",
        "has_restaurant", "has_parking", "has_breakfast", "has_kitchen",
        "has_bathtub", "is_free_cancellation",
        "has_pool_view", "has_garden_view", "has_ocean_view2",
        "has_private_pool", "has_pool_access", "has_nature_view", "has_city_view",
    ]

    # Prefer same kecamatan
    subset = hotel_df[hotel_df["kecamatan_cleaned"].str.lower() == kecamatan.lower()].copy()
    if len(subset) < n:
        subset = hotel_df.copy()

    # Also match by room_tier proximity
    tier = input_dict.get("room_tier", 1)
    subset["tier_diff"] = abs(subset["room_tier"] - tier)

    user_vec = np.array([input_dict.get(c, 0) for c in flag_cols], dtype=float)
    existing = subset[flag_cols].values.astype(float)

    # Cosine similarity
    norms = np.linalg.norm(existing, axis=1)
    user_norm = np.linalg.norm(user_vec)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_sim = np.where(
            (norms > 0) & (user_norm > 0),
            existing @ user_vec / (norms * user_norm),
            0.0,
        )

    # Combined score: 70% cosine sim + 30% tier match
    tier_score = 1.0 / (1.0 + subset["tier_diff"].values)
    combined   = 0.7 * cos_sim + 0.3 * tier_score

    subset = subset.copy()
    subset["_score"] = combined

    # Deduplicate: keep best-scoring room per hotel name
    subset = subset.sort_values("_score", ascending=False)
    subset = subset.drop_duplicates(subset=["name"], keep="first")
    top = subset.head(n)

    results = []
    for _, row in top.iterrows():
        results.append({
            "name":             row.get("name", "—"),
            "kecamatan":        row.get("kecamatan_cleaned", "—"),
            "property_type":    row.get("property_type", "—"),
            "star_rating":      int(row.get("star_rating", 0)),
            "rating":           float(row.get("rating", 0)),
            "review_count":     int(row.get("review_count", 0)),
            "room_name":        row.get("room_name", "—"),
            "room_tier":        int(row.get("room_tier", 1)),
            "price_idr":        int(row.get("price_numeric", 0)),
            "similarity_score": round(float(row["_score"]), 4),
        })
    return results


# ===========================================================================
# HELPER — Similar Properties for Kos
# ===========================================================================
def get_similar_kos(features: dict, location: str, n: int = 5) -> List[Dict]:
    """Find similar kos listings using amenity cosine similarity."""
    subset = kos_df[kos_df["location"].str.lower() == location.lower()].copy()
    if len(subset) < n:
        subset = kos_df.copy()

    user_vec = np.array([features.get(c, 0) for c in ALL_AMENITIES], dtype=float)
    existing = subset[ALL_AMENITIES].values.astype(float)

    norms = np.linalg.norm(existing, axis=1)
    user_norm = np.linalg.norm(user_vec)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_sim = np.where(
            (norms > 0) & (user_norm > 0),
            existing @ user_vec / (norms * user_norm),
            0.0,
        )

    subset = subset.copy()
    subset["_score"] = cos_sim

    # Deduplicate by name
    subset = subset.sort_values("_score", ascending=False)
    subset = subset.drop_duplicates(subset=["name"], keep="first")
    top = subset.head(n)

    results = []
    for _, row in top.iterrows():
        amenities_present = [a for a in ALL_AMENITIES if row.get(a, 0) == 1]
        results.append({
            "name":             row.get("name", "—"),
            "location":         row.get("location", "—"),
            "price_idr":        int(row.get("price", 0)),
            "amenity_count":    int(row.get("amenity_count", 0)),
            "is_luxury":        int(row.get("is_luxury", 0)),
            "amenities":        amenities_present,
            "similarity_score": round(float(row["_score"]), 4),
        })
    return results


# ===========================================================================
# ENDPOINTS — HEALTH
# ===========================================================================
@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "models": {
            "hotel_pernight": "loaded",
            "kos_permonth":   "loaded",
        },
        "datasets": {
            "hotel_rows": len(hotel_df),
            "kos_rows":   len(kos_df),
        },
    }


# ===========================================================================
# ENDPOINTS — METADATA
# ===========================================================================
@app.get("/metadata/pernight", tags=["Metadata"])
def metadata_pernight():
    """Return all dropdown / selection options for the hotel/villa prediction form."""
    return HOTEL_METADATA


@app.get("/metadata/permonth", tags=["Metadata"])
def metadata_permonth():
    """Return all dropdown / selection options for the kos prediction form."""
    return KOS_METADATA


# ===========================================================================
# ENDPOINTS — PREDICT HOTEL (PER NIGHT)
# ===========================================================================
@app.post("/predict/pernight", tags=["Prediction"])
def predict_pernight(data: HotelPredictInput):
    """
    Predict nightly room price for a hotel or villa in Bali.

    Returns:
    - **prediction**: estimated IDR price + confidence range (±36% MAPE)
    - **similar_properties**: up to `similar_count` comparable listings from training data
    """
    try:
        # ---------------------------------------------------------
        # Pre-process / map frontend fields to backend model fields
        # ---------------------------------------------------------
        if data.is_luxury is not None:
            data.is_luxury_brand = data.is_luxury

        if data.facilities:
            facs = [f.lower() for f in data.facilities]
            if "pool" in facs or "swimming pool" in facs: data.has_pool = 1
            if "wifi" in facs: data.has_wifi = 1
            if "ac" in facs or "air conditioning" in facs: data.has_ac = 1
            if "gym" in facs or "fitness" in facs: data.has_gym = 1
            if "spa" in facs: data.has_spa = 1
            if "restaurant" in facs: data.has_restaurant = 1
            if "parking" in facs: data.has_parking = 1
            if "breakfast" in facs: data.has_breakfast = 1
            if "kitchen" in facs: data.has_kitchen = 1
            if "bathtub" in facs: data.has_bathtub = 1
            if "free cancellation" in facs: data.is_free_cancellation = 1

        if data.exclusive_feature:
            exf = data.exclusive_feature.lower()
            if "private pool" in exf: data.has_private_pool = 1
            elif "ocean" in exf or "sea" in exf or "beach" in exf: data.has_ocean_view2 = 1
            elif "pool view" in exf: data.has_pool_view = 1
            elif "garden" in exf: data.has_garden_view = 1
            elif "pool access" in exf: data.has_pool_access = 1
            elif "nature" in exf: data.has_nature_view = 1
            elif "city" in exf: data.has_city_view = 1

        input_dict = {
            "log_review":         np.log1p(data.review_count),
            "star_rating":        data.star_rating,
            "rating":             data.rating,
            "no_review":          1 if data.rating == 0 else 0,
            "room_size_m2":       data.room_size_m2 if data.room_size_m2 else hotel_preproc["room_size_m2_median"],
            "latitude":           data.latitude,
            "longitude":          data.longitude,
            "is_luxury_brand":    data.is_luxury_brand,
            "property_type_enc":  data.property_type_enc,
            "has_pool":           data.has_pool,
            "has_wifi":           data.has_wifi,
            "has_ac":             data.has_ac,
            "has_gym":            data.has_gym,
            "has_spa":            data.has_spa,
            "has_restaurant":     data.has_restaurant,
            "has_parking":        data.has_parking,
            "has_breakfast":      data.has_breakfast,
            "has_kitchen":        data.has_kitchen,
            "has_bathtub":        data.has_bathtub,
            "has_balcony":        0,    # always 0 in training data
            "has_ocean_view":     0,    # always 0 in training data
            "is_free_cancellation": data.is_free_cancellation,
            "room_tier":          data.room_tier,
            "value_added_count":  data.value_added_count,
            "has_pool_view":      data.has_pool_view,
            "has_garden_view":    data.has_garden_view,
            "has_ocean_view2":    data.has_ocean_view2,
            "has_private_pool":   data.has_private_pool,
            "has_pool_access":    data.has_pool_access,
            "has_nature_view":    data.has_nature_view,
            "has_city_view":      data.has_city_view,
            "kecamatan_cleaned":  data.kecamatan,
        }

        X = pd.DataFrame([input_dict])
        X["kecamatan_te"] = hotel_preproc["target_encoder"].transform(
            X[["kecamatan_cleaned"]]
        )["kecamatan_cleaned"]

        dmat  = xgb.DMatrix(X[hotel_preproc["feature_cols"]])
        log_p = hotel_model.predict(dmat)[0]
        
        # Workaround for XGBoost JSON bug where base_score is dropped on some Linux environments
        if log_p < 5.0:
            log_p += (13.693568 - 0.5)
            
        price = int(np.expm1(log_p))

        # Confidence range based on model MAPE (~36%)
        margin_pct = 0.36
        margin     = int(price * margin_pct)

        similar = get_similar_hotels(input_dict, data.kecamatan, data.similar_count)

        return {
            "status": "success",
            "type":   "per_night",
            "prediction": {
                "estimated_price_idr": price,
                "range_min_idr":       max(0, price - margin),
                "range_max_idr":       price + margin,
                "currency":            "IDR",
                "note":                "Model MAPE ~36%. Range covers ±36% uncertainty.",
            },
            "similar_properties": similar,
        }

    except Exception as exc:
        logger.exception("predict_pernight error")
        raise HTTPException(status_code=500, detail=str(exc))


# ===========================================================================
# ENDPOINTS — PREDICT KOS (PER MONTH)
# ===========================================================================
@app.post("/predict/permonth", tags=["Prediction"])
def predict_permonth(data: KosPredictInput):
    """
    Predict monthly kos (room rental) price in Bali.

    Returns:
    - **prediction**: estimated IDR price + confidence range
    - **similar_properties**: up to `similar_count` comparable kos listings
    """
    try:
        loc = data.location.lower().strip()
        if loc not in KOS_LOCATION_MAP:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unknown location '{data.location}'. "
                    f"Valid values: {sorted(KOS_LOCATION_MAP.keys())}"
                ),
            )

        features = compute_kos_features(data)
        X        = pd.DataFrame([features])

        # Ensure column order matches training
        kos_feature_order = [
            "ac", "air_panas", "kamar_mandi_dalam", "parkir", "dispenser",
            "kulkas", "kursi", "lemari", "meja", "parkir_mobil", "ruang_tamu",
            "shower", "tv", "wastafel", "wifi", "cctv", "swimming_pool",
            "kasur_bed_springbed",
            "amenity_count", "premium_count", "amenity_density",
            "has_private_bath_ac", "luxury_score", "is_luxury",
            "location_tier", "location_enc",
        ]
        X = X[kos_feature_order]

        price = float(kos_model.predict(X)[0])
        price = max(0, int(round(price)))

        # Confidence range — use ±25% as a reasonable estimate
        margin_pct = 0.25
        margin     = int(price * margin_pct)

        similar = get_similar_kos(features, data.location, data.similar_count)

        return {
            "status": "success",
            "type":   "per_month",
            "prediction": {
                "estimated_price_idr": price,
                "range_min_idr":       max(0, price - margin),
                "range_max_idr":       price + margin,
                "currency":            "IDR",
                "note":                "Estimated ±25% confidence range.",
            },
            "similar_properties": similar,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("predict_permonth error")
        raise HTTPException(status_code=500, detail=str(exc))


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
