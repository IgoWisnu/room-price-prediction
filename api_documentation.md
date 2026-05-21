# Bali Rent Price Prediction API Documentation

This documentation provides details on the core prediction endpoints: `/predict/pernight` and `/predict/permonth`. Both endpoints require POST requests with JSON payloads and return a predicted price alongside similar properties found in the training dataset.

---

## 1. Predict Hotel / Villa (Per Night)
**Endpoint:** `POST /predict/pernight`

This endpoint predicts the nightly room rate for a hotel, villa, or similar property. **It has been updated to easily accept arrays of strings for facilities** directly from your frontend.

### Example Request Body
```json
{
  "hotel_name": "Contoh Hotel Bali",
  "kecamatan": "Kuta",
  "star_rating": 4,
  "rating": 8.5,
  "review_count": 150,
  "latitude": -8.721122,
  "longitude": 115.109533,
  "room_size_m2": 45.0,
  "is_luxury": 0,
  "room_tier": 2,
  "property_type_enc": 4,
  "exclusive_feature": "Private Pool",
  "facilities": [
    "AC",
    "WiFi",
    "Breakfast",
    "Pool",
    "gym",
    "spa"
  ],
  "similar_count": 5
}
```

### Field Definitions (Hotel/Villa)
- `hotel_name`: (String) Display name.
- `kecamatan`: (String) Region/Sub-district (e.g., "Kuta", "Seminyak", "Ubud").
- `star_rating`: (Integer) Star rating (0-5).
- `rating`: (Float) Guest review score (0.0 to 10.0). Use `0.0` if none.
- `review_count`: (Integer) Total reviews.
- `latitude` / `longitude`: (Float) GPS coordinates.
- `room_size_m2`: (Float) Room size in square meters.
- `is_luxury`: (Integer) `1` if it belongs to a luxury brand chain, else `0`.
- `room_tier`: (Integer) `1` (Standard) to `5` (Villa/Penthouse).
- `property_type_enc`: (Integer) See `/metadata/pernight` for mapping (e.g., `4` = Hotel).
- `exclusive_feature`: (String) Special feature like `"Private Pool"`, `"Ocean View"`, `"City View"`.
- `facilities`: (Array of Strings) Any combination of `"AC"`, `"WiFi"`, `"Pool"`, `"Gym"`, `"Spa"`, `"Restaurant"`, `"Parking"`, `"Breakfast"`, `"Kitchen"`, `"Bathtub"`.
- `similar_count`: (Integer) Number of comparable recommendations to return (default: 5).

### Example Response
```json
{
  "status": "success",
  "type": "per_night",
  "prediction": {
    "estimated_price_idr": 1250000,
    "range_min_idr": 800000,
    "range_max_idr": 1700000,
    "currency": "IDR",
    "note": "Model MAPE ~36%. Range covers ±36% uncertainty."
  },
  "similar_properties": [
    {
      "name": "Kuta Lagoon Resort",
      "kecamatan": "Kuta",
      "property_type": "Hotel",
      "star_rating": 4,
      "rating": 8.4,
      "review_count": 210,
      "room_name": "Deluxe Pool Access",
      "room_tier": 2,
      "price_idr": 1200000,
      "similarity_score": 0.954
    }
  ]
}
```

---

## 2. Predict Kos / Room Rental (Per Month)
**Endpoint:** `POST /predict/permonth`

This endpoint predicts the monthly rate for a Kos (boarding house). Unlike the Hotel endpoint, it expects an explicitly defined `1` or `0` for each amenity.

### Example Request Body
```json
{
  "location": "denpasar",
  "ac": 1,
  "air_panas": 1,
  "kamar_mandi_dalam": 1,
  "parkir": 1,
  "dispenser": 0,
  "kulkas": 1,
  "kursi": 1,
  "lemari": 1,
  "meja": 1,
  "parkir_mobil": 0,
  "ruang_tamu": 0,
  "shower": 1,
  "tv": 1,
  "wastafel": 0,
  "wifi": 1,
  "cctv": 0,
  "swimming_pool": 0,
  "kasur_bed_springbed": 1,
  "similar_count": 5
}
```

### Field Definitions (Kos)
- `location`: (String) Supported areas like `"denpasar"`, `"badung"`, `"jimbaran"`. Check `/metadata/permonth` for full list.
- `ac` through `kasur_bed_springbed`: (Integer) Set to `1` if the Kos provides the amenity, or `0` if not.
- `similar_count`: (Integer) Number of comparable Kos recommendations to return (default: 5).

### Example Response
```json
{
  "status": "success",
  "type": "per_month",
  "prediction": {
    "estimated_price_idr": 1850000,
    "range_min_idr": 1387500,
    "range_max_idr": 2312500,
    "currency": "IDR",
    "note": "Estimated ±25% confidence range."
  },
  "similar_properties": [
    {
      "name": "Kos Exclusive Tukad Barito",
      "location": "denpasar",
      "price_idr": 1800000,
      "amenity_count": 12,
      "is_luxury": 0,
      "amenities": [
        "ac",
        "air_panas",
        "kamar_mandi_dalam",
        "wifi",
        "kasur_bed_springbed"
      ],
      "similarity_score": 0.981
    }
  ]
}
```
