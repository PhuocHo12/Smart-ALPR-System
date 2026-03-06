# 🚗 ALPR System (YOLO + FastALPR)

A **config-driven Automatic License Plate Recognition (ALPR)** system built for **computer vision portfolios and production-ready demos**.

This project combines:
- **Vehicle detection** using YOLO (Ultralytics)
- **License plate detection + OCR** using FastALPR
- **YAML-based configuration** (no hardcoded parameters)
- **Modern Python tooling with `uv`**
- **Docker-first workflow** (CPU-ready, GPU-friendly)

---

## ✨ Features

- 🚙 Vehicle detection: `car`, `motorcycle`, `bus`, `truck`
- 🔍 License plate detection (end-to-end)
- 🔤 OCR for plate text
- ⚙️ Configurable via YAML (`yolo.yaml`, `alpr.yaml`)
- 📦 Reproducible builds with `uv.lock`
- 🐳 Dockerized (no pip, no venv pain)

---

## 📁 Project Structure

```
project/
├── src/
│   ├── detectors/
│   │   └── vehicle_detector.py
│   ├── recognizers/
│   │   └── fastalpr_engine.py
│   ├── pipeline/
│   │   └── alpr_pipeline.py
│   └── main.py
├── configs/
│   ├── yolo.yaml
│   └── alpr.yaml
├── models/
│   └── yolo26n.pt
├── assets/
│   └── demo.mp4
├── pyproject.toml
├── uv.lock
├── Dockerfile
└── README.md
```
---

## ⚙️ Configuration

### `configs/yolo.yaml`

```yaml
model:
  path: models/yolov8n.pt
  conf: 0.4
  device: cpu

detection:
  class_map:
    car: 2
    motorcycle: 3
    bus: 5
    truck: 7
```

### `configs/alpr.yaml`

```yaml
alpr:
  detector_model: yolo-v9-t-384-license-plate-end2end
  ocr_model: cct-xs-v1-global-model
  threshold: 0.6
```

---
## 🎬 Demo

Quick Preview

A short demo showing vehicle detection + license plate recognition running end-to-end.

👉 ▶️ Watch demo video

![ALPR Demo](demo.gif)

## 🚀 Getting Started (Local)

### 1️⃣ Install dependencies with `uv`

```bash
uv sync
```

### 2️⃣ Run the demo

```bash
uv run python src/main.py
```

---

## 🐳 Docker Usage (CPU)

### Build image

```bash
docker build -t alpr-system .
```

### Run container

```bash
docker run --rm \
  -v $(pwd)/models:/app/models \
  alpr-system
```

> Mounting `models/` allows swapping YOLO weights without rebuilding the image.

---

## 🧠 Design Highlights

- **Config-driven** → easy experiments & deployment
- **Separated concerns** (vehicle detection vs ALPR)
- **Production-minded** (Docker + lockfile)
- **Portfolio-friendly** (clear architecture)

---

## 🛣️ Roadmap

- [ ] Vehicle → plate ROI chaining
- [ ] Object tracking (ByteTrack / DeepSORT)
- [ ] FastAPI inference service
- [ ] RTSP / multi-camera pipeline
- [ ] GPU Docker profile (CUDA)
- [ ] Benchmarking (FPS / latency)

---

## ⚠️ Notes

- Default setup runs on **CPU**
- GPU support requires CUDA-based Docker image
- Tested on **Linux**


## 👤 Author

Built for showcasing **Computer Vision & AI Engineering** skills.

If you find this useful ⭐ the repo or fork it for your own experiments.
