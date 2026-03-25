from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import math
import os
import json

app = FastAPI(title="Humanitarian Logistics API")

# Connect to a local Redis instance (needs to be running on the machine)
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True, socket_timeout=1)
    redis_client.ping()
except Exception as e:
    redis_client = None
    print("Redis not connected. Queue features will fail.")

class DeliveryEvent(BaseModel):
    device_id: str
    latitude: float
    longitude: float
    timestamp: str
    image_hash: str # Hash of the captured photo to prove it's uniquely taken

# Haversine formula implemented on the backend as requested in Step 7
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Radius of Earth in kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    
    a = (math.sin(dLat/2) * math.sin(dLat/2) + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dLon/2) * math.sin(dLon/2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

@app.post("/verify-delivery")
async def verify_delivery(event: DeliveryEvent):
    # Dummy target location (e.g., NGO warehouse coordinates, SRMIST campus)
    target_lat = 12.8236
    target_lon = 80.0435
    
    distance_km = haversine(target_lat, target_lon, event.latitude, event.longitude)
    
    # Reject if claiming to be at the location but actually > 0.5km away (Spoofing block)
    if distance_km > 0.5:
        if redis_client:
            redis_client.lpush("failed_deliveries", json.dumps(event.dict()))
        raise HTTPException(status_code=403, detail=f"Location validation failed. Hardware is {distance_km:.2f}km away from target.")
        
    # Valid delivery location
    if redis_client:
        redis_client.lpush("verified_deliveries", json.dumps(event.dict()))
        
    return {"status": "success", "message": "Delivery location verified via Haversine and queued.", "distance_km": distance_km}

from fastapi.responses import FileResponse

@app.get("/")
def read_root():
    return FileResponse("src/static/index.html")

@app.get("/data/{folder}/{filename}")
def get_image(folder: str, filename: str):
    return FileResponse(f"data/{folder}/{filename}")
