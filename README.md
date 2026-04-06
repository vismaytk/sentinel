<div align="center">

# 🛡️ SENTINEL

**Vehicle Intelligence Platform**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](#license)

*A production-grade, Palantir-style defence intelligence system for real-time vehicle detection, tracking, and license plate recognition.*

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [API](#api-endpoints) • [Configuration](#configuration-options)

</div>

---

## 🎯 Overview

SENTINEL provides a sophisticated web-based dashboard for monitoring vehicle activity through IP cameras, with advanced features like multi-object tracking, OCR-based plate reading, zone monitoring, and comprehensive analytics.

<div align="center">
<img src="https://img.shields.io/badge/Live%20Dashboard-Real--time%20Detection-00d4ff?style=for-the-badge" alt="Live Dashboard"/>
<img src="https://img.shields.io/badge/Analytics-Charts%20%26%20Insights-00ff88?style=for-the-badge" alt="Analytics"/>
<img src="https://img.shields.io/badge/Threat%20Level-Automated%20Assessment-ff4444?style=for-the-badge" alt="Threat Level"/>
</div>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SENTINEL PLATFORM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   CAMERA    │───▶│  DETECTOR   │───▶│   TRACKER   │───▶│     OCR     │  │
│  │   (IP/USB)  │    │  (YOLOv8)   │    │   (SORT)    │    │  (EasyOCR)  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         DETECTION WORKER                            │   │
│  │   • Decoupled Pipeline  • Adaptive Frame Skip  • Zone Monitoring   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                     │             │
│         ▼                                                     ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   MJPEG     │    │   ALERTS    │    │  SNAPSHOTS  │    │   SQLite    │  │
│  │   STREAM    │    │   ENGINE    │    │   MANAGER   │    │   DATABASE  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                  │                  │          │
│         └──────────────────┴──────────────────┴──────────────────┘          │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         WEB DASHBOARD                               │   │
│  │   • Live Feed  • Alerts  • Analytics  • Reports  • Configuration   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎥 **Live Video Feed** | Real-time MJPEG stream with detection overlays |
| 🚗 **Vehicle Detection** | YOLOv8-powered detection of commercial & military vehicles |
| 🔢 **Object Tracking** | SORT algorithm with persistent track IDs and confirmation |
| 📝 **Plate OCR** | EasyOCR-based license plate text recognition |
| 🚨 **Alert System** | Configurable rules with cooldown and severity levels |
| 🗺️ **Zone Monitoring** | User-defined polygonal zones with entry detection |
| 📸 **Auto Snapshots** | Automatic capture on military detection and plate reads |
| 📊 **Analytics Dashboard** | Chart.js visualizations, heatmaps, and insights |
| 📄 **Intelligence Reports** | Server-rendered printable reports |
| ⚡ **Real-time Push** | Server-Sent Events (SSE) for instant updates |
| 🗄️ **SQLite Storage** | Persistent detection history with WAL mode & batch writes |
| ⚙️ **Live Config** | Adjust confidence thresholds without restart |
| 🎨 **Palantir UI** | Dark-themed, professional defence aesthetic |

### New in v2.0

- **Decoupled Detection Pipeline** — Streaming and detection run in separate threads for smoother video
- **Adaptive Frame Skip** — Automatically adjusts detection frequency based on system load
- **Test-Time Augmentation (TTA)** — Optional accuracy boost with WBF merge
- **Multi-Scale Inference** — Detect vehicles at multiple scales for better accuracy
- **Platt Scaling** — Calibrated confidence scores per vehicle class
- **Track Confirmation** — Reduces false positives by requiring multiple detections
- **TurboJPEG Support** — Optional faster JPEG encoding

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- YOLO model files (see below)

### 1. Clone the Repository

```bash
git clone https://github.com/vismaytk/sentinel.git
cd sentinel
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Optional performance dependencies:
```bash
# Faster JPEG encoding (recommended)
pip install PyTurboJPEG
```

### 3. Download Models

Place these YOLOv8 model files in the `models/` directory:

| Model | Description | Required |
|-------|-------------|----------|
| `vehicle.pt` | Vehicle detection (commercial & military) | ✅ Yes |
| `license_plate.pt` | License plate detection | ✅ Yes |

> ⚠️ **Note**: Model files are not included in the repository due to size. Contact the maintainer or train your own models.

### 4. Configure Camera

Edit `config.py` to set your IP camera URL:
```python
IP_CAM_URL = "http://192.168.31.27:8080/video"
```

Supported sources:
- IP Camera: `http://192.168.x.x:8080/video`
- RTSP Stream: `rtsp://user:pass@192.168.x.x/stream`
- Webcam: `"0"` (device index as string)
- Video File: `"/path/to/video.mp4"`

---

## 🚀 Usage

### Starting the Server

```bash
python app.py
```

The server will display a startup banner and be available at:

| Endpoint | URL |
|----------|-----|
| 🖥️ Dashboard | http://localhost:5000 |
| 📊 Analytics | http://localhost:5000/analytics |
| 📄 Report | http://localhost:5000/report |
| 💚 Health Check | http://localhost:5000/health |

### Detection Pipeline

```
Camera → CLAHE Enhancement → YOLOv8 Detection → SORT Tracking → OCR → Zones → Snapshots → Database
```

1. **Capture** — Frame from IP camera with reconnection logic
2. **Enhance** — CLAHE preprocessing for varied lighting
3. **Detect** — Two-stage YOLO (vehicle → plate crop) with optional TTA
4. **Track** — SORT algorithm assigns persistent IDs with confirmation
5. **OCR** — EasyOCR with caching, throttling, and parallel workers
6. **Zones** — Check detections against user-defined monitoring zones
7. **Snapshots** — Auto-capture on military vehicles and plate reads
8. **Store** — SQLite with async background batch writer

---

## ⚙️ Configuration Options

All settings are in `config.py`:

| Option | Default | Description |
|--------|---------|-------------|
| `IP_CAM_URL` | `"http://192.168.31.27:8080/video"` | Camera source URL |
| `PORT` | `5000` | Web server port |
| `YOLO_IMGSZ` | `480` | YOLO inference size (320/416/480/640) |
| `DETECT_EVERY_N` | `2` | Run detection every N frames |
| `ENABLE_OCR` | `True` | Enable plate text recognition |
| `ENABLE_TRACKING` | `True` | Enable vehicle tracking |
| `ENABLE_TTA` | `False` | Enable test-time augmentation |
| `MULTI_SCALE_INFERENCE` | `False` | Run detection at two scales |
| `TRACK_MAX_AGE` | `30` | Frames before track is dropped |
| `TRACK_MIN_HITS` | `3` | Hits required to confirm track |
| `OCR_WORKERS` | `2` | Parallel OCR worker threads |
| `class_conf["commercial-vehicle"]` | `0.70` | Commercial vehicle confidence |
| `class_conf["military_vehicle"]` | `0.30` | Military vehicle confidence |
| `plate_conf["License_Plate"]` | `0.30` | Plate detection confidence |

> 💡 **Tip**: Confidence thresholds can be adjusted live via the dashboard config panel.

---

## 🔌 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Live dashboard |
| `/analytics` | GET | Analytics page |
| `/report` | GET | Intelligence report page |
| `/video_feed` | GET | MJPEG video stream |
| `/stream` | GET | SSE real-time stats |
| `/stats` | GET | JSON stats snapshot |
| `/config` | GET/POST | Get/update config |
| `/health` | GET | Health check |
| `/api/health/db` | GET | Database integrity check |

### Detection & Analytics

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detections` | GET | Detection history (paginated) |
| `/api/analytics` | GET | Analytics aggregates |
| `/api/sessions` | GET | Session list |
| `/api/clear` | POST | Clear old data |

### Alert System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/alerts` | GET | Get active alerts |
| `/api/alerts/clear` | POST | Clear all alerts |
| `/api/alerts/<id>` | DELETE | Dismiss specific alert |
| `/api/alerts/rules` | GET | Get alert rules |

### Zone Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/zones` | GET | List all zones |
| `/api/zones` | POST | Create new zone |
| `/api/zones/<id>` | GET/PUT/DELETE | Manage specific zone |
| `/api/zones/events` | GET | Zone entry events |
| `/api/zones/occupancy` | GET | Current zone occupancy |

### Snapshots

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/snapshots` | GET | List snapshots |
| `/api/snapshots/capture` | POST | Manual capture |
| `/api/snapshots/<id>` | GET/DELETE | Manage snapshot |
| `/api/snapshots/image/<id>` | GET | Full image |
| `/api/snapshots/thumbnail/<id>` | GET | Thumbnail |
| `/api/snapshots/stats` | GET | Snapshot statistics |

### Example: Get Detections

```bash
curl "http://localhost:5000/api/detections?limit=50&type=military_vehicle"
```

### Example: Create Zone

```bash
curl -X POST "http://localhost:5000/api/zones" \
  -H "Content-Type: application/json" \
  -d '{"name": "Entry Gate", "polygon": [[100,100],[300,100],[300,300],[100,300]]}'
```

---

## 📁 Directory Structure

```
sentinel/
├── core/                   # Core detection modules
│   ├── __init__.py
│   ├── camera.py          # Camera stream with reconnection
│   ├── detector.py        # YOLOv8 two-stage pipeline with TTA
│   ├── tracker.py         # SORT tracking with confirmation
│   ├── ocr.py             # EasyOCR with caching & parallel workers
│   ├── database.py        # SQLite with WAL mode & batch writes
│   ├── alerts.py          # Alert engine with configurable rules
│   ├── zones.py           # Zone monitoring system
│   ├── snapshots.py       # Auto-capture snapshot manager
│   └── validate.py        # Startup validation checks
├── api/                    # Flask API layer
│   ├── __init__.py        # App factory
│   ├── routes.py          # All HTTP endpoints + DetectionWorker
│   └── events.py          # Server-Sent Events
├── static/
│   ├── css/
│   │   └── sentinel.css   # Palantir dark theme
│   └── js/
│       ├── dashboard.js   # Live feed + alerts
│       ├── analytics.js   # Chart.js integration
│       └── config.js      # Config panel logic
├── templates/
│   ├── base.html          # Shared layout
│   ├── dashboard.html     # Live feed page
│   ├── analytics.html     # Analytics page
│   └── report.html        # Intelligence report
├── models/                 # YOLO model files (.gitignore'd)
│   └── .gitkeep
├── data/                   # SQLite database & snapshots (runtime)
│   └── .gitkeep
├── config/                 # Zone configurations (runtime)
├── logs/                   # Log files
│   └── .gitkeep
├── config.py              # Centralized configuration
├── app.py                 # Entry point
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

---

## ⚠️ Known Limitations

| Limitation | Details |
|------------|---------|
| **Single Camera** | Currently supports one camera source at a time |
| **OCR Accuracy** | Works best with clear, unobstructed plates |
| **Browser Support** | SSE requires modern browsers (no IE) |
| **Memory Usage** | EasyOCR uses ~500MB RAM when active |
| **Model Files** | Not included in repo (download separately) |

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, Flask 3.0
- **ML/CV**: YOLOv8 (Ultralytics), OpenCV, EasyOCR
- **Frontend**: Vanilla JS, Chart.js, CSS3
- **Database**: SQLite with WAL mode
- **Real-time**: Server-Sent Events (SSE)

---

## 📄 License

Proprietary — For authorized use only.

---

<div align="center">

**Built with ❤️ for Defence Intelligence**

</div>
