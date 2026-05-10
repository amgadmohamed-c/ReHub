# ReHub — AI-Powered Wearable Health Monitoring Backend

> **FastAPI · PostgreSQL · scikit-learn · WebSockets**  
> Real-time exercise classification and injury-risk prediction for Arduino wearables.

---

## Architecture

```
Arduino Wearable
      │  (BLE / Serial)
      ▼
Mobile App (Flutter / React Native)
      │  POST /api/v1/sensors/{device_id}/readings
      ▼
┌─────────────────────────────────────────────────────┐
│                  FastAPI Backend                    │
│                                                     │
│  ┌──────────────┐    ┌──────────────────────────┐  │
│  │ Sliding      │    │  ML Pipelines            │  │
│  │ Window (50)  │───▶│  • Exercise Classifier   │  │
│  │ Buffer       │    │  • Injury Predictor      │  │
│  └──────────────┘    └──────────┬───────────────┘  │
│                                 │                   │
│  ┌──────────────┐    ┌──────────▼───────────────┐  │
│  │ PostgreSQL   │◀───│  Alert Engine            │  │
│  │ (readings,   │    │  (thresholds + severity) │  │
│  │  analyses,   │    └──────────────────────────┘  │
│  │  alerts)     │                                   │
│  └──────────────┘                                   │
│                                                     │
│  WebSocket (/ws/{device_id}) ──────────────────────▶ Mobile App
└─────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Details |
|---|---|
| **Exercise Classification** | walking · running · push-up · sit-up · standing · resting |
| **Injury Prediction** | fatigue · strain risk · injury risk · overexertion (scores 0–1) |
| **Sliding Window** | Configurable window size (default 50) + step (default 10) |
| **Real-time WebSocket** | Push analysis results and alerts instantly to mobile app |
| **Alert Engine** | Configurable thresholds, severity levels (LOW/MEDIUM/HIGH/CRITICAL) |
| **Async throughout** | FastAPI + asyncpg + SQLAlchemy async |
| **Swagger UI** | Full API docs at `/docs` |

---

## Sensor Payload

The mobile app sends this JSON payload (from the Arduino wearable) to the backend:

```json
{
  "emg":    114,
  "flex":   1023,
  "flow":   0,
  "ax":     0.299275,
  "ay":    -1.482011,
  "az":    11.02051,
  "gx":    -0.083403,
  "gy":    -0.022649,
  "gz":     0.012257,
  "temp":  30.03588
}
```

Movement intensity is computed server-side: **I = √(ax² + ay² + az²)**

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/amgadmohamed-c/ReHub.git
cd ReHub
cp .env.example .env
# Edit .env with your database credentials
```

### 2. Docker Compose (recommended)

```bash
# Start PostgreSQL + API
docker compose up -d

# Train ML models (first time only)
docker compose --profile train up trainer

# View logs
docker compose logs -f api
```

The API is now running at **http://localhost:8000**  
Swagger docs: **http://localhost:8000/docs**

---

### 3. Local Development

**Prerequisites:** Python 3.11+, PostgreSQL 14+

```bash
# Create virtualenv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure database
createdb rehub
# Edit DATABASE_URL in .env

# Train ML models
python ml_training/train_models.py

# Run migrations (optional — init_db auto-creates tables on startup)
alembic upgrade head

# Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## API Reference

### Sensor Ingestion

```http
POST /api/v1/sensors/{device_id}/readings
Content-Type: application/json

{
  "emg": 114, "flex": 1023, "flow": 0,
  "ax": 0.299275, "ay": -1.482011, "az": 11.02051,
  "gx": -0.083403, "gy": -0.022649, "gz": 0.012257,
  "temp": 30.03588
}
```

**Response (202 Accepted):**
```json
{
  "reading_id": 42,
  "device_id": "arduino-001",
  "movement_intensity": 11.12,
  "buffer_size": 23,
  "analysis_triggered": false,
  "analysis": null,
  "alerts": []
}
```

When `buffer_size` reaches 50 an analysis is triggered:
```json
{
  "reading_id": 92,
  "device_id": "arduino-001",
  "movement_intensity": 9.87,
  "buffer_size": 40,
  "analysis_triggered": true,
  "analysis": {
    "id": 1,
    "exercise_type": "walking",
    "exercise_confidence": 0.82,
    "fatigue_score": 0.12,
    "strain_risk": 0.08,
    "injury_risk": 0.05,
    "overexertion_score": 0.09,
    "avg_movement_intensity": 10.21,
    "avg_temp": 36.7
  },
  "alerts": []
}
```

---

### Analysis History

```http
GET /api/v1/analysis/{device_id}/history?limit=10&offset=0
GET /api/v1/analysis/{device_id}/latest
```

---

### Alerts

```http
GET /api/v1/alerts/{device_id}?unacknowledged_only=true
POST /api/v1/alerts/{device_id}/acknowledge
Body: { "alert_ids": [1, 2, 3] }
```

---

### Health Check

```http
GET /health

{
  "status": "ok",
  "version": "1.0.0",
  "db": "ok",
  "models_loaded": true
}
```

---

### Device Registration

```http
POST /api/v1/devices
{ "device_id": "arduino-001", "name": "Amgad's Wearable" }
```

---

## WebSocket Real-Time Alerts

Connect from the mobile app:

```
ws://localhost:8000/ws/{device_id}
```

**Flutter example:**
```dart
final channel = WebSocketChannel.connect(
  Uri.parse('ws://your-server/ws/arduino-001'),
);
channel.stream.listen((message) {
  final data = jsonDecode(message);
  if (data['event'] == 'analysis') {
    final analysis = data['analysis'];
    // Update UI with exercise type, scores, alerts
  }
});
```

**Server push events:**

| event | When | Payload |
|---|---|---|
| `connected` | On connection | Welcome message |
| `analysis` | After each window | Full analysis + alerts |
| `heartbeat` | Every 30 s | Timestamp |

**Client events:**

| event | Action |
|---|---|
| `ping` | Server replies `pong` |

---

## ML Pipelines

### Exercise Classifier (RandomForest)
- **Input:** 36 features from accelerometer + gyroscope window
- **Features:** mean, std, min, max, range per axis + zero crossings + intensity stats
- **Output:** exercise label + confidence score (0–1)

### Injury Predictor (Gradient Boosting)
- **Input:** 30 features from EMG, flex, flow, temperature, movement intensity
- **Output:** 4 independent risk scores (fatigue, strain, injury, overexertion)

### Alert Thresholds (configurable via .env)

| Metric | Default threshold |
|---|---|
| Fatigue | 0.65 |
| Strain risk | 0.60 |
| Injury risk | 0.55 |
| Overexertion | 0.70 |
| Temperature | 38.5 °C |

---

## Project Structure

```
rehub/
├── app/
│   ├── core/
│   │   ├── config.py         # Pydantic settings
│   │   └── logging.py        # Logging setup
│   ├── database/
│   │   ├── models.py         # SQLAlchemy ORM models
│   │   └── session.py        # Async engine + session factory
│   ├── ml/
│   │   ├── model_registry.py # Load + inference interface
│   │   └── models/           # .pkl files (generated by training)
│   ├── routes/
│   │   ├── sensor_routes.py  # POST /sensors
│   │   ├── analysis_routes.py
│   │   ├── alert_routes.py
│   │   ├── ws_routes.py      # WebSocket
│   │   └── health_routes.py
│   ├── schemas/
│   │   └── schemas.py        # Pydantic v2 request/response models
│   ├── services/
│   │   ├── sensor_service.py # Core ingestion orchestrator
│   │   └── alert_service.py  # Alert generation
│   ├── utils/
│   │   ├── features.py       # Feature engineering
│   │   ├── window_buffer.py  # Sliding window buffer
│   │   └── ws_manager.py     # WebSocket connection manager
│   └── main.py               # FastAPI app factory
├── ml_training/
│   ├── train_models.py       # Generate data + train + save models
│   └── datasets/             # Generated CSVs
├── alembic/                  # DB migrations
├── scripts/
│   └── api_example.py        # Full usage example
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Running the Example Script

```bash
# In one terminal, start the server
uvicorn app.main:app --reload

# In another terminal
python scripts/api_example.py
```

---

## Mobile App Integration (Flutter / React Native)

The backend is fully CORS-enabled for mobile development.

**Base URL:** `http://<your-server>:8000/api/v1`  
**WebSocket:** `ws://<your-server>:8000/ws/<device_id>`

Key integration points:
1. Call `POST /devices` once on app launch to register the device
2. Send readings continuously via `POST /sensors/{device_id}/readings`
3. Listen on the WebSocket for real-time analysis + alerts
4. Poll `GET /alerts/{device_id}?unacknowledged_only=true` for missed alerts
5. Acknowledge viewed alerts via `POST /alerts/{device_id}/acknowledge`
