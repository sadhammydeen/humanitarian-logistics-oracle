# Humanitarian Logistics Verification Platform

A secure, automated verification system for in-kind humanitarian donations using lightweight computer vision and geospatial validation. Designed to operate on standard mobile devices in resource-constrained NGO environments.

## 🏗️ Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Visual Verification** | MobileNetV2 (~3.5M params) | Classify donations as Food, Clothing, or Medicine |
| **Low-Light Enhancement** | CLAHE (LAB L* channel) | Improve classification accuracy in dark warehouses |
| **Geospatial Validation** | Haversine Formula (50m threshold) | Verify delivery location via great-circle distance |
| **Cryptographic Integrity** | SHA-256 Hash | Non-repudiation of verification events |
| **Verification Metric** | Transparency Score | `TS = 0.5 × geo_score + 0.5 × visual_score` |
| **Backend** | FastAPI + Redis | Event-driven API with queuing |
| **Mobile Client** | React Native | High-accuracy GPS extraction |

---

## 🚀 Getting Started

```bash
# From the project root
cd /Users/sadhammydeen/Documents/humanitarian_logistics/

# Launch the backend (installs deps + starts FastAPI)
./start_backend.sh
```

Navigate to `http://localhost:8000` for the interactive verification dashboard.

---

## 📂 Project Structure

### 🧠 Core Platform (`src/`)
* **`main.py`** — The FastAPI backend. Handles the great-circle Haversine GPS formula, the cryptographic SHA-256 transparency hashing, and the endpoints (`/verify-delivery`, `/classify-image`, `/enhance-clahe`).
* **`mobilenet_classifier.py`** — The MobileNetV2 inference engine. Extremely fast (<3.5s latency), built specifically for edge processors, and maps ImageNet features to Food/Clothing/Medicine.
* **`clahe_enhancer.py`** — The Computer Vision pre-processor. Applies Contrast Limited Adaptive Histogram Equalization directly to the L* (lightness) channel to brighten dark warehouse / night-time photos without destroying original colors.
* **`train_mobilenet.py`** — Main script to train the MobileNetV2 pipeline on a full cloud dataset with augmentation.
* **`static/index.html`** — The Mission-critical dashboard frontend UI.
* **`mobile/LocationScanner.js`** — React Native extraction of high-accuracy GPS coordinates at the delivery site.

### 🛠️ Data & Training Scripts (`scripts/`)
* **`download_dataset.py`** — Scrapes humanitarian images from Wikimedia Commons for massive model training.
* **`create_subset.py`** — Creates perfectly balanced subsets (e.g. 5,000 Food vs 500 Background) for highly efficient, 20-minute cloud GPU training.
* **`train_demo.py`** — Fast, single-epoch demo training script to test hardware pipelines without heavy downloads.

### 🔬 Research & Analytics (`experiments/`)
* **`experiment_a_vision.py`** — Validates the accuracy boost of our CLAHE model against poor lighting conditions.
* **`experiment_c_efficiency.py`** — Tracks our sub-3.5 second latency benchmark across 50 simulated real-world donation events.

### 🧪 Validation (`tests/`)
* **`test_spatial.py`** — Simulates an attacker trying to spoof GPS coordinates 51m away, proving the Haversine firewall successfully blocks it.

---

## 📊 Key Metrics
- **Spatial Security:** 100% block rate against simulated GPS spoofing attacks
- **Latency Target:** Sub-3.5 second end-to-end processing per verification
- **Accuracy Target:** ≥88% classification accuracy with CLAHE preprocessing
- **Model Size:** ~3.5M parameters (edge-deployable on mobile CPU)
