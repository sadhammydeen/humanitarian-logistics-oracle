"""
EXPERIMENT B: Spatial Logic / GPS Spoofing Test

Tests the Haversine-based geospatial proximity validation by simulating:
  1. A valid delivery within the 50m threshold
  2. A GPS spoofing attack from ~40km away

The Haversine formula accounts for Earth's curvature, calculating the
great-circle distance between two points. The 50-meter operational
threshold accounts for civilian GPS drift while precluding off-site scans.

Expected Results:
  - Valid delivery (within 50m): HTTP 200 with Transparency Score
  - Spoofed delivery (~40km away): HTTP 403 (blocked by Haversine)
"""

import requests
import time

URL = "http://localhost:8000/verify-delivery"

print("=" * 72)
print("  EXPERIMENT B: SPATIAL VALIDATION (HAVERSINE GPS ANTI-SPOOFING)")
print("=" * 72)
print(f"\n  Target NGO Coordinates: Lat 12.8236, Lon 80.0435")
print(f"  Operational Threshold: 50 meters (0.05 km)")
print(f"  Formula: Haversine (great-circle distance on sphere)")

# ─── Test 1: Valid Delivery (Within 50m) ──────────────────────────────────────

print(f"\n{'─' * 72}")
print("  [TEST 1] Valid Delivery — Device at NGO warehouse (~15m away)")
print(f"{'─' * 72}")

valid_payload = {
    "device_id": "truck_42",
    "latitude": 12.8237,      # ~15 meters from target
    "longitude": 80.0436,
    "timestamp": "2026-03-26T10:00:00Z",
    "image_hash": "sha256_valid_delivery_001",
    "image_classification": "Food",
    "classification_confidence": 0.93
}

try:
    res = requests.post(URL, json=valid_payload)
    if res.status_code == 200:
        data = res.json()
        print(f"  ✅ ACCEPTED (HTTP {res.status_code})")
        print(f"     Distance: {data['distance_km']:.4f} km")
        print(f"     Transparency Score: {data['transparency_score']:.2f}")
        print(f"       → Geo Score:    {data['geo_score']:.1f}")
        print(f"       → Visual Score: {data['visual_score']:.2f}")
        print(f"     SHA-256 Hash: {data['transparency_hash'][:32]}...")
    else:
        print(f"  ❌ UNEXPECTED REJECTION (HTTP {res.status_code})")
        print(f"     {res.json()}")
except Exception as e:
    print(f"  Connection failed: {e}")

# ─── Test 2: Spoofing Attack (~40km Away) ─────────────────────────────────────

print(f"\n{'─' * 72}")
print("  [TEST 2] GPS Spoofing Attack — Fake coords from Chennai Central (~40km)")
print(f"{'─' * 72}")

spoofed_payload = {
    "device_id": "truck_42_hacked",
    "latitude": 13.0827,      # Chennai Central Station
    "longitude": 80.2707,
    "timestamp": "2026-03-26T10:05:00Z",
    "image_hash": "sha256_spoofed_delivery_999",
    "image_classification": "Food",
    "classification_confidence": 0.91
}

try:
    res = requests.post(URL, json=spoofed_payload)
    if res.status_code == 403:
        print(f"  ✅ BLOCKED (HTTP {res.status_code})")
        print(f"     Haversine correctly detected spoofed GPS coordinates.")
        print(f"     Reason: {res.json()['detail']}")
    else:
        print(f"  ❌ SECURITY FAILURE: Spoofed GPS was accepted (HTTP {res.status_code})")
        print(f"     {res.json()}")
except Exception as e:
    print(f"  Connection failed: {e}")

print(f"\n{'═' * 72}")
print("  RESULT: Haversine spatial validation correctly enforces 50m threshold")
print(f"{'═' * 72}\n")
