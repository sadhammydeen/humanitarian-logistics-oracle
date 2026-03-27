"""
EXPERIMENT C: System Efficiency and Latency

Measures end-to-end latency across 50 trials to confirm the sub-3.5-second
processing target per request. Additionally models efficiency gains by comparing
the automated workflow against traditional manual NGO verification processes,
which typically involve human review and delays spanning up to 24 hours.

This comparative analysis is crucial for justifying the adoption of the technology
to NGO administrators, demonstrating that AI verification reduces the
administrative burden significantly.

Metrics tracked:
  - Per-request latency (ms)
  - Mean, median, p95, max latency
  - Total processing time for 50 trials
  - Efficiency gain vs manual verification (24-hour baseline)
"""

import time
import requests
import csv
import os
import sys
import statistics

# ─── Configuration ────────────────────────────────────────────────────────────

URL = "http://localhost:8000/verify-delivery"
TOTAL_TRIALS = 50  # Per tech doc specification
TARGET_LATENCY_S = 3.5  # Sub-3.5 second processing target
RESULTS_DIR = "data/results"

# Manual verification baseline: traditional NGO processes involve human review
# with delays spanning up to 24 hours per the tech doc
MANUAL_VERIFICATION_DELAY_HOURS = 24
MANUAL_TOTAL_SECONDS = MANUAL_VERIFICATION_DELAY_HOURS * 3600  # 86,400 seconds


def run_experiment():
    print("=" * 72)
    print("  EXPERIMENT C: SYSTEM EFFICIENCY & LATENCY")
    print(f"  {TOTAL_TRIALS} Trials | Target: <{TARGET_LATENCY_S}s per request")
    print("=" * 72)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ─── Phase 1: Automated API Benchmark ─────────────────────────────────

    print(f"\n  Booting Automated Event-Driven API Benchmark...")
    print(f"  Endpoint: {URL}")
    print(f"  Method: Haversine geospatial validation + Transparency Score\n")

    latencies = []
    successful = 0
    failed = 0

    total_start = time.time()

    for i in range(TOTAL_TRIALS):
        # Simulate a convoy of delivery trucks near the NGO target warehouse
        payload = {
            "device_id": f"convoy_truck_{i:03d}",
            "latitude": 12.8236 + (i * 0.000005),  # Micro-variations within 50m
            "longitude": 80.0435 + (i * 0.000003),
            "timestamp": f"2026-03-26T12:{i:02d}:00Z",
            "image_hash": f"sha256_delivery_{i:03d}",
            "image_classification": "Food",
            "classification_confidence": 0.92
        }

        req_start = time.time()
        try:
            res = requests.post(URL, json=payload, timeout=10)
            req_end = time.time()
            latency_ms = (req_end - req_start) * 1000
            latencies.append(latency_ms)

            if res.status_code == 200:
                successful += 1
                data = res.json()
                if i < 3 or i == TOTAL_TRIALS - 1:
                    print(f"  [Trial {i+1:2d}/{TOTAL_TRIALS}] ✅ {latency_ms:7.1f}ms | "
                          f"TS={data.get('transparency_score', 'N/A'):.2f} | "
                          f"Hash={data.get('transparency_hash', 'N/A')[:16]}...")
                elif i == 3:
                    print(f"  ... running {TOTAL_TRIALS - 4} more trials ...")
            else:
                failed += 1
                if i < 3:
                    print(f"  [Trial {i+1:2d}/{TOTAL_TRIALS}] ❌ {latency_ms:7.1f}ms | "
                          f"Status: {res.status_code}")

        except requests.exceptions.ConnectionError:
            print(f"\n  ERROR: Cannot connect to {URL}")
            print(f"  Ensure the FastAPI server is running:")
            print(f"    python -m uvicorn src.main:app --port 8000")
            sys.exit(1)
        except Exception as e:
            req_end = time.time()
            latency_ms = (req_end - req_start) * 1000
            latencies.append(latency_ms)
            failed += 1

    total_end = time.time()
    total_api_time = total_end - total_start

    # ─── Phase 2: Statistical Analysis ────────────────────────────────────

    print(f"\n{'═' * 72}")
    print("  RESULTS & DATA FOR RESEARCH PAPER")
    print(f"{'═' * 72}")

    if latencies:
        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        stdev_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0

        # P95 latency
        sorted_latencies = sorted(latencies)
        p95_index = int(len(sorted_latencies) * 0.95)
        p95_latency = sorted_latencies[min(p95_index, len(sorted_latencies) - 1)]

        print(f"\n  {'Metric':<45s} {'Value':>15s}")
        print(f"  {'─' * 60}")
        print(f"  {'Total Trials':<45s} {TOTAL_TRIALS:>15d}")
        print(f"  {'Successful Transactions':<45s} {successful:>12d}/{TOTAL_TRIALS}")
        print(f"  {'Failed Transactions':<45s} {failed:>12d}/{TOTAL_TRIALS}")
        print(f"  {'Total API Processing Time':<45s} {total_api_time:>13.2f}s")
        print(f"  {'Mean Latency per Request':<45s} {mean_latency:>12.1f}ms")
        print(f"  {'Median Latency per Request':<45s} {median_latency:>12.1f}ms")
        print(f"  {'P95 Latency':<45s} {p95_latency:>12.1f}ms")
        print(f"  {'Max Latency':<45s} {max_latency:>12.1f}ms")
        print(f"  {'Min Latency':<45s} {min_latency:>12.1f}ms")
        print(f"  {'Std Deviation':<45s} {stdev_latency:>12.1f}ms")

        # Sub-3.5s target check (converting to ms for per-request)
        target_ms = TARGET_LATENCY_S * 1000
        below_target = sum(1 for l in latencies if l < target_ms)
        print(f"\n  {'Requests under target (<3.5s)':<45s} {below_target:>12d}/{len(latencies)}")

        if mean_latency < target_ms:
            print(f"\n  ✅ LATENCY TARGET MET: Mean {mean_latency:.1f}ms < {target_ms:.0f}ms ({TARGET_LATENCY_S}s)")
        else:
            print(f"\n  ❌ LATENCY TARGET NOT MET: Mean {mean_latency:.1f}ms >= {target_ms:.0f}ms")

        # ─── Efficiency Comparison vs Manual NGO Verification ─────────

        print(f"\n  {'─' * 60}")
        print(f"  EFFICIENCY COMPARISON: Automated vs Manual NGO Verification")
        print(f"  {'─' * 60}")
        print(f"  {'Manual NGO Process (human review)':<45s} {MANUAL_TOTAL_SECONDS:>12.0f}s ({MANUAL_VERIFICATION_DELAY_HOURS}h)")
        print(f"  {'Automated API Processing':<45s} {total_api_time:>13.2f}s")

        if total_api_time > 0:
            efficiency_gain = MANUAL_TOTAL_SECONDS / total_api_time
            time_saved = MANUAL_TOTAL_SECONDS - total_api_time
            print(f"\n  => EFFICIENCY GAIN: {efficiency_gain:,.0f}x faster than manual verification")
            print(f"  => TIME SAVED: {time_saved/3600:.2f} hours of administrative overhead")
            print(f"  => The automated system processed {TOTAL_TRIALS} deliveries in")
            print(f"     {total_api_time:.2f}s vs ~{MANUAL_VERIFICATION_DELAY_HOURS}h for manual review")

    # ─── Save Latency Data to CSV ─────────────────────────────────────────

    results_csv = os.path.join(RESULTS_DIR, "experiment_c_latency.csv")
    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Trial", "Latency_ms", "Status"])
        for i, latency in enumerate(latencies):
            status = "success" if i < successful else "failed"
            writer.writerow([i + 1, f"{latency:.2f}", status])

    # Summary row
    summary_csv = os.path.join(RESULTS_DIR, "experiment_c_summary.csv")
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        if latencies:
            writer.writerow(["Total_Trials", TOTAL_TRIALS])
            writer.writerow(["Successful", successful])
            writer.writerow(["Total_API_Time_s", f"{total_api_time:.4f}"])
            writer.writerow(["Mean_Latency_ms", f"{mean_latency:.2f}"])
            writer.writerow(["Median_Latency_ms", f"{median_latency:.2f}"])
            writer.writerow(["P95_Latency_ms", f"{p95_latency:.2f}"])
            writer.writerow(["Max_Latency_ms", f"{max_latency:.2f}"])
            writer.writerow(["Manual_Baseline_s", MANUAL_TOTAL_SECONDS])
            writer.writerow(["Efficiency_Gain_x", f"{efficiency_gain:.0f}"])

    print(f"\n  Latency data saved to: {results_csv}")
    print(f"  Summary saved to: {summary_csv}")
    print(f"{'═' * 72}\n")


if __name__ == "__main__":
    run_experiment()
