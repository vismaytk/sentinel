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

SENTINEL provides a sophisticated web-based dashboard for monitoring vehicle activity through IP cameras, with advanced features like multi-object tracking, OCR-based plate reading, and comprehensive analytics.

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
│  │                         SHARED STATE                                │   │
│  │   • Detection Results  • Track IDs  • Plate Text  • Statistics     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                     │             │
│         ▼                                                     ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   MJPEG     │    │     SSE     │    │    REST     │    │   SQLite    │  │
│  │   STREAM    │    │   EVENTS    │    │     API     │    │   DATABASE  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                  │                  │          │
│         └──────────────────┴──────────────────┴──────────────────┘          │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         WEB DASHBOARD                               │   │
│  │   • Live Video Feed  • Threat Log  • Analytics  • Configuration    │   │
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
| 🔢 **Object Tracking** | SORT algorithm with persistent track IDs |
| 📝 **Plate OCR** | EasyOCR-based license plate text recognition |
| 📊 **Analytics Dashboard** | Chart.js visualizations, heatmaps, and insights |
| ⚡ **Real-time Push** | Server-Sent Events (SSE) for instant updates |
| 🗄️ **SQLite Storage** | Persistent detection history with WAL mode |
| ⚙️ **Live Config** | Adjust confidence thresholds without restart |
| 🎨 **Palantir UI** | Dark-themed, professional defence aesthetic |

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
| 💚 Health Check | http://localhost:5000/health |

### Detection Pipeline

```
Camera → CLAHE Enhancement → YOLOv8 Detection → SORT Tracking → OCR → Database
```

1. **Capture** — Frame from IP camera with reconnection logic
2. **Enhance** — CLAHE preprocessing for varied lighting
3. **Detect** — Two-stage YOLO (vehicle → plate crop)
4. **Track** — SORT algorithm assigns persistent IDs
5. **OCR** — EasyOCR with caching and throttling
6. **Store** — SQLite with async background writer

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
| `TRACK_MAX_AGE` | `30` | Frames before track is dropped |
| `class_conf["commercial-vehicle"]` | `0.70` | Commercial vehicle confidence |
| `class_conf["military_vehicle"]` | `0.30` | Military vehicle confidence |
| `plate_conf["License_Plate"]` | `0.30` | Plate detection confidence |

> 💡 **Tip**: Confidence thresholds can be adjusted live via the dashboard config panel.

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Live dashboard |
| `/analytics` | GET | Analytics page |
| `/video_feed` | GET | MJPEG video stream |
| `/stream` | GET | SSE real-time stats |
| `/stats` | GET | JSON stats snapshot |
| `/config` | GET/POST | Get/update config |
| `/api/detections` | GET | Detection history (paginated) |
| `/api/analytics` | GET | Analytics aggregates |
| `/api/sessions` | GET | Session list |
| `/api/clear` | POST | Clear old data |
| `/health` | GET | Health check |

### Example: Get Detections

```bash
curl "http://localhost:5000/api/detections?limit=50&type=military_vehicle"
```

---

## 📁 Directory Structure

```
sentinel/
├── core/                   # Core detection modules
│   ├── __init__.py
│   ├── camera.py          # Camera stream with reconnection
│   ├── detector.py        # YOLOv8 two-stage pipeline
│   ├── tracker.py         # SORT tracking algorithm
│   ├── ocr.py             # EasyOCR with caching
│   └── database.py        # SQLite with WAL mode
├── api/                    # Flask API layer
│   ├── __init__.py        # App factory
│   ├── routes.py          # All HTTP endpoints
│   └── events.py          # Server-Sent Events
├── static/
│   ├── css/
│   │   └── sentinel.css   # Palantir dark theme
│   └── js/
│       ├── dashboard.js   # Live feed + sparklines
│       ├── analytics.js   # Chart.js integration
│       └── config.js      # Config panel logic
├── templates/
│   ├── base.html          # Shared layout
│   ├── dashboard.html     # Live feed page
│   └── analytics.html     # Analytics page
├── models/                 # YOLO model files (.gitignore'd)
│   └── .gitkeep
├── data/                   # SQLite database (runtime)
│   └── .gitkeep
├── logs/                   # Log files
│   └── .gitkeep
├── config.py              # Centralized configuration
├── app.py                 # Entry point (~90 lines)
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
