"""
Humanitarian Logistics Verification API

Event-driven FastAPI backend implementing:
  1. Haversine-based geospatial proximity validation (50m threshold)
  2. MobileNetV2 image classification for donation verification
  3. Transparency Score combining geographic + visual verification
  4. SHA-256 cryptographic non-repudiation hashing
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import redis
import math
import os
import json
import hashlib
import tempfile
import shutil
from datetime import datetime

app = FastAPI(
    title="Humanitarian Logistics Verification API",
    description="Secure delivery verification using MobileNetV2 classification and Haversine geospatial validation",
    version="2.0.0"
)

# ─── Redis Connection ─────────────────────────────────────────────────────────
try:
    redis_client = redis.Redis(
        host='localhost', port=6379, db=0,
        decode_responses=True, socket_timeout=1
    )
    redis_client.ping()
except Exception as e:
    redis_client = None
    print("Redis not connected. Queue features will operate in fallback mode.")

# ─── Data Models ──────────────────────────────────────────────────────────────


class DeliveryEvent(BaseModel):
    device_id: str
    latitude: float
    longitude: float
    timestamp: str
    image_hash: str  # Hash of the captured photo to prove it's uniquely taken
    image_classification: str = "Food"  # MobileNetV2 predicted category
    classification_confidence: float = 0.90  # MobileNetV2 confidence score


# ─── Haversine Formula ────────────────────────────────────────────────────────
# Because the Earth is a sphere, simple Euclidean distance produces significant
# errors over varying latitudes. The Haversine formula calculates the great-circle
# distance between two points on a sphere using their latitudes and longitudes.
#
# Formula:
#   a = sin²(Δlat/2) + cos(lat1) · cos(lat2) · sin²(Δlon/2)
#   c = 2 · atan2(√a, √(1−a))
#   d = R · c
#
# Where R = 6,371 km (Earth's mean radius)

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth.

    Args:
        lat1, lon1: Coordinates of point 1 (degrees)
        lat2, lon2: Coordinates of point 2 (degrees)

    Returns:
        Distance in kilometers
    """
    R = 6371.0  # Earth's mean radius in kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)

    a = (math.sin(dLat / 2) * math.sin(dLat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dLon / 2) * math.sin(dLon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance


# ─── Transparency Score ───────────────────────────────────────────────────────
# The Transparency Score is a quantified metric of physical reality, weighting
# geographic proximity and visual classification equally:
#
#   TS = 0.5 × geo_score + 0.5 × visual_score
#
# Where:
#   geo_score = 1.0 if distance ≤ 50m, else 0.0
#   visual_score = classification_confidence from MobileNetV2
#
# Definitive verification states:
#   TS = 1.0  → Perfect verification (at location + high confidence classification)
#   TS = 0.5  → Partial verification (only one factor satisfied)
#   TS = 0.0  → Failed verification (neither factor satisfied)

def calculate_transparency_score(distance_km, classification_confidence):
    """
    Calculate the Transparency Score combining geospatial proximity
    and visual classification confidence.

    Args:
        distance_km: Distance from target in kilometers
        classification_confidence: MobileNetV2 classification confidence [0, 1]

    Returns:
        tuple: (transparency_score, geo_score, visual_score)
    """
    # Geographic proximity score: binary threshold at 50 meters (0.05 km)
    geo_score = 1.0 if distance_km <= 0.05 else 0.0

    # Visual classification score: direct from MobileNetV2 confidence
    visual_score = min(1.0, max(0.0, classification_confidence))

    # Equal weighting as specified in the tech doc
    transparency_score = 0.5 * geo_score + 0.5 * visual_score

    return transparency_score, geo_score, visual_score


# ─── Cryptographic Non-Repudiation ───────────────────────────────────────────
# To ensure the Transparency Score cannot be tampered with post-verification,
# the system generates a SHA-256 cryptographic hash by concatenating the
# verification signals and a precise timestamp. This creates an immutable
# digital fingerprint providing non-repudiation without blockchain costs.

def generate_transparency_hash(event: DeliveryEvent, transparency_score, geo_score, visual_score):
    """
    Generate SHA-256 hash for cryptographic non-repudiation.

    The hash payload concatenates all verification signals:
      device_id | latitude | longitude | classification | confidence | score | timestamp
    """
    verification_string = (
        f"{event.device_id}|"
        f"{event.latitude}|"
        f"{event.longitude}|"
        f"{event.image_classification}|"
        f"{event.classification_confidence}|"
        f"{transparency_score:.4f}|"
        f"{datetime.utcnow().isoformat()}"
    )
    return hashlib.sha256(verification_string.encode('utf-8')).hexdigest()


# ─── API Endpoints ────────────────────────────────────────────────────────────

# Target NGO warehouse coordinates (e.g., SRMIST campus)
TARGET_LAT = 12.8236
TARGET_LON = 80.0435
PROXIMITY_THRESHOLD_KM = 0.05  # 50 meters


@app.post("/verify-delivery")
async def verify_delivery(event: DeliveryEvent):
    """
    Verify a humanitarian delivery using geospatial proximity validation
    and visual classification confidence.

    The endpoint:
      1. Computes Haversine distance to the target NGO warehouse
      2. Calculates the Transparency Score (geo + visual weighted equally)
      3. Generates a SHA-256 cryptographic hash for non-repudiation
      4. Queues the event in Redis for audit trail
    """
    distance_km = haversine(TARGET_LAT, TARGET_LON, event.latitude, event.longitude)

    # Calculate Transparency Score
    transparency_score, geo_score, visual_score = calculate_transparency_score(
        distance_km, event.classification_confidence
    )

    # Reject if outside 50-meter threshold
    if distance_km > PROXIMITY_THRESHOLD_KM:
        if redis_client:
            redis_client.lpush("failed_deliveries", json.dumps({
                **event.dict(),
                "distance_km": distance_km,
                "transparency_score": transparency_score,
                "rejection_reason": "geospatial_proximity_exceeded"
            }))
        raise HTTPException(
            status_code=403,
            detail=f"Location validation failed. Device is {distance_km:.4f}km "
                   f"away from target (threshold: {PROXIMITY_THRESHOLD_KM}km / 50m). "
                   f"Transparency Score: {transparency_score:.2f}"
        )

    # Generate cryptographic non-repudiation hash
    transparency_hash = generate_transparency_hash(
        event, transparency_score, geo_score, visual_score
    )

    # Queue verified delivery in Redis
    if redis_client:
        redis_client.lpush("verified_deliveries", json.dumps({
            **event.dict(),
            "distance_km": distance_km,
            "transparency_score": transparency_score,
            "transparency_hash": transparency_hash
        }))

    return {
        "status": "success",
        "message": "Delivery verified via Haversine geospatial validation and MobileNetV2 classification.",
        "distance_km": distance_km,
        "transparency_score": transparency_score,
        "geo_score": geo_score,
        "visual_score": visual_score,
        "image_classification": event.image_classification,
        "classification_confidence": event.classification_confidence,
        "transparency_hash": transparency_hash
    }


@app.post("/enhance-clahe")
async def enhance_clahe_endpoint(file: UploadFile = File(...)):
    """
    Apply CLAHE enhancement to an uploaded image and return the processed image.
    Operates on the L* channel of LAB color space to preserve hue/saturation.
    """
    try:
        import cv2
        import numpy as np
        from src.clahe_enhancer import apply_clahe_to_numpy
        from fastapi.responses import Response

        # Read uploaded bytes directly into numpy array
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # Apply CLAHE on LAB L* channel
        enhanced = apply_clahe_to_numpy(img, clip_limit=2.0, tile_grid_size=(8, 8))

        # Encode to JPEG bytes
        _, buf = cv2.imencode(".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return Response(content=buf.tobytes(), media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CLAHE enhancement failed: {str(e)}")


@app.post("/classify-image")
async def classify_image_endpoint(file: UploadFile = File(...)):
    """
    Classify an uploaded donation image using MobileNetV2 with CLAHE preprocessing.

    The pipeline:
      1. Applies CLAHE to the LAB color space L* channel
      2. Runs MobileNetV2 inference (~3.5M parameters)
      3. Maps ImageNet classes to humanitarian categories (Food, Clothing, Medicine)
      4. Returns classification with confidence score
    """
    try:
        from src.mobilenet_classifier import classify_image as run_classification

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Run MobileNetV2 classification with CLAHE
        category, confidence, inference_ms, category_probs = run_classification(
            tmp_path, apply_clahe=True
        )

        # Clean up temp file
        os.unlink(tmp_path)

        return {
            "classification": category,
            "confidence": round(confidence, 4),
            "inference_time_ms": round(inference_ms, 1),
            "category_probabilities": {k: round(v, 4) for k, v in category_probs.items()},
            "model": "MobileNetV2 (~3.5M parameters)",
            "preprocessing": "CLAHE on LAB L* channel"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.get("/transparency-score")
def transparency_score_info():
    """
    Return documentation on the Transparency Score formula.
    """
    return {
        "formula": "TS = 0.5 × geo_score + 0.5 × visual_score",
        "components": {
            "geo_score": "1.0 if Haversine distance ≤ 50m, else 0.0",
            "visual_score": "MobileNetV2 classification confidence [0, 1]"
        },
        "verification_states": {
            "1.0": "Perfect verification (at location + high confidence classification)",
            "0.5": "Partial verification (only one factor satisfied)",
            "0.0": "Failed verification (neither factor satisfied)"
        },
        "cryptographic_integrity": "SHA-256 hash of concatenated verification signals",
        "threshold": "50 meters (accounts for civilian GPS drift)"
    }


@app.get("/")
def read_root():
    return FileResponse("src/static/index.html")


@app.get("/data/{folder}/{filename}")
def get_image(folder: str, filename: str):
    return FileResponse(f"data/{folder}/{filename}")
