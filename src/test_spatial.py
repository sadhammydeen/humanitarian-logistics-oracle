import requests
import time

URL = "http://localhost:8000/verify-delivery"

print("--- EXPERIMENT B: THE HACKER TEST (SPATIAL LOGIC) ---")
print("Target Node Coordinates in Backend: Lat 12.8236, Lon 80.0435 (SRMIST)")
print("Allowed Radius: 0.5km")

# 1. Valid Delivery Attempt (Same coordinates)
print("\n[TEST 1] Testing Valid Delivery Node Device (Right at the warehouse)...")
valid_payload = {
    "device_id": "truck_42",
    "latitude": 12.8237, # Just ~15 meters away
    "longitude": 80.0436,
    "timestamp": "2026-03-26T10:00:00Z",
    "image_hash": "hash_valid_123"
}
try:
    res = requests.post(URL, json=valid_payload)
    if res.status_code == 200:
        print("✅ SUCCESS. Valid delivery accepted.", res.json())
    else:
        print("❌ FAILED. Valid delivery rejected.", res.status_code, res.json())
except Exception as e:
    print("Failed to connect to backend", e)

# 2. Spoofing Attempt (Different city/coordinates)
# Example: 13.0827, 80.2707 (Chennai Central Station - ~40km away)
print("\n[TEST 2] Testing Fake GPS Spoofing Attack from hacker (~40km away)...")
spoofed_payload = {
    "device_id": "truck_42_hacked",
    "latitude": 13.0827,
    "longitude": 80.2707,
    "timestamp": "2026-03-26T10:05:00Z",
    "image_hash": "hash_spoofed_999"
}
try:
    res = requests.post(URL, json=spoofed_payload)
    if res.status_code == 403:
        print("✅ SUCCESS. Hacker's Fake GPS blocked correctly by Haversine equation.", res.status_code, res.json())
    else:
        print("❌ FAILED. Hacker bypassed security.", res.status_code, res.json())
except Exception as e:
    print("Failed to connect to backend", e)
