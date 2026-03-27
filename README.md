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

### Data Engineering & Computer Vision
* **`src/mobilenet_classifier.py`**
  MobileNetV2 classification module using depthwise separable convolutions and inverted residual blocks. Maps ImageNet-1K classes to humanitarian categories (Food, Clothing, Medicine) with integrated CLAHE preprocessing. Completes inference in <5 seconds on mobile CPU.

* **`src/clahe_enhancer.py`**
  Contrast Limited Adaptive Histogram Equalization applied exclusively to the L* (lightness) channel of the LAB color space. Enhances contrast while preserving original hue and saturation — preventing color distortion that would confuse CNN classification.

* **`src/download_dataset.py`**
  Automated dataset acquisition via Wikimedia Commons API for humanitarian aid images.

### Spatial Oracle Backend
* **`src/main.py`**
  Event-driven FastAPI server implementing:
  - **Haversine formula** for great-circle distance calculation (50m threshold)
  - **Transparency Score** = 0.5 × geo_score + 0.5 × visual_score
  - **SHA-256 cryptographic hash** for non-repudiation of verification events
  - **`/classify-image`** endpoint for real-time MobileNetV2 donation classification

* **`src/mobile/LocationScanner.js`**
  React Native client extracting high-accuracy GPS coordinates for delivery verification.

### Experiments
* **`src/experiment_a_vision.py`** — Tests CLAHE efficacy: baseline vs CLAHE-enhanced MobileNetV2 accuracy with confusion matrices and Mean Average Precision tracking (target: ≥88%).

* **`src/test_spatial.py`** — GPS spoofing attack simulation proving 100% block rate via Haversine validation.

* **`src/experiment_c_efficiency.py`** — End-to-end latency benchmark across 50 trials targeting sub-3.5s processing, compared against 24-hour manual NGO verification baseline.

---

## 📊 Key Metrics
- **Spatial Security:** 100% block rate against simulated GPS spoofing attacks
- **Latency Target:** Sub-3.5 second end-to-end processing per verification
- **Accuracy Target:** ≥88% classification accuracy with CLAHE preprocessing
- **Model Size:** ~3.5M parameters (edge-deployable on mobile CPU)
