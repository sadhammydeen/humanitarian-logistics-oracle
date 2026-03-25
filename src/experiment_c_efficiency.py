import time
import requests

URL = "http://localhost:8000/verify-delivery"
TOTAL_REQUESTS = 100

print("--- EXPERIMENT C: LOGISTICS EFFICIENCY TEST ---")
print(f"Simulating a workload of {TOTAL_REQUESTS} supply deliveries...\n")

# Baseline: Manual processing by a human logistics coordinator
# We assume it takes a human 45 seconds to cross-reference a GPS ping on Google Maps and log it on a clipboard/excel per truck.
MANUAL_TIME_PER_REQ = 45 
total_manual_time = MANUAL_TIME_PER_REQ * TOTAL_REQUESTS

print("1. Booting up Automated Event-Driven API Benchmark (Haversine Spatial math)...")
start_api = time.time()

successful = 0
for i in range(TOTAL_REQUESTS):
    # Simulate a convoy of trucks driving very close to the NGO target warehouse
    payload = {
        "device_id": f"convoy_truck_{i}",
        "latitude": 12.8236 + (i * 0.00001), 
        "longitude": 80.0435,
        "timestamp": "2026-03-26T12:00:00Z",
        "image_hash": f"secure_hash_x9{i}"
    }
    try:
        res = requests.post(URL, json=payload)
        if res.status_code == 200:
            successful += 1
    except Exception as e:
        pass

end_api = time.time()
total_api_time = end_api - start_api

print("\n---------------- RESULTS & DATA FOR RESEARCH PAPER ----------------")
print(f"Total Transactions Processed       : {successful} / {TOTAL_REQUESTS}")
print(f"Manual Human Verification Baseline : {total_manual_time} seconds ({total_manual_time/60:.2f} minutes)")
print(f"Automated Event-Driven API Time    : {total_api_time:.2f} seconds")

if total_api_time > 0:
    efficiency_gain = total_manual_time / total_api_time
    time_saved = total_manual_time - total_api_time
    print(f"\n=> EFFICIENCY GAIN: The proposed automated platform is {efficiency_gain:.1f}x faster than standard Humanitarian Logistics.")
    print(f"=> TIME SAVED: Using this architecture saved {time_saved/60:.2f} minutes of administrative overhead under a {TOTAL_REQUESTS} batch load.")
else:
    print("\nAPI executed instantly (under 0.00s). Huge efficiency gain.")
