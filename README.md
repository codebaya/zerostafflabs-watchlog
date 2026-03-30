# WatchLog — Week 1 MVP

AI-powered CCTV monitoring for small businesses.

**Stack**: Python · FastAPI · ffmpeg · YOLO · Qwen VL · Telegram Bot · SQLite

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env: set TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, QWEN_BACKEND

# 3. Run the API server
python main.py
# → http://localhost:8000

# 4. (Separate terminal) Run the camera watcher
python watcher.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check |
| GET | `/cameras` | List all cameras |
| POST | `/cameras` | Add a camera |
| GET | `/cameras/{id}` | Camera detail |
| POST | `/cameras/{id}/probe` | Test RTSP connection |
| GET | `/alerts` | List alerts (optional `?camera_id=`) |
| POST | `/alerts/{id}/feedback` | Submit false-positive feedback |
| POST | `/telegram/webhook` | Telegram bot webhook |

## Add Your First Camera

```bash
curl -X POST http://localhost:8000/cameras \
  -H "Content-Type: application/json" \
  -d '{"name": "입구 카메라", "rtsp_url": "rtsp://192.168.1.100:554/stream", "location": "1층 입구"}'
```

## AI Backends

Set `QWEN_BACKEND` in `.env`:

| Value | Requires | Notes |
|-------|----------|-------|
| `mock` | nothing | Random results, for testing |
| `ollama` | [Ollama](https://ollama.com) running locally + `ollama pull qwen2.5vl:7b` | Recommended for Mac Mini |
| `transformers` | `pip install transformers torch` | Direct HuggingFace load |

YOLO uses `ultralytics` — install with `pip install ultralytics`.
Model defaults to `yolo11n.pt` (auto-downloaded on first run).

## Run Tests

```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

## Project Structure

```
watchlog/
├── main.py              # FastAPI app
├── watcher.py           # Real-time RTSP watcher loop
├── config.py            # Settings (pydantic-settings)
├── rtsp/
│   ├── client.py        # ffmpeg/OpenCV RTSP client
│   └── frame_extractor.py
├── ai/
│   ├── classifier.py    # Qwen VL anomaly classifier
│   ├── detector.py      # YOLO object detector
│   └── pipeline.py      # Combined pipeline
├── notifications/
│   └── telegram.py      # Telegram alert sender
├── api/
│   ├── models.py        # Pydantic schemas
│   └── routes/          # FastAPI routers
└── storage/
    └── database.py      # SQLAlchemy async SQLite
```
