"""API endpoint tests (no external dependencies required)."""
import os
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_watchlog.db")
os.environ.setdefault("QWEN_BACKEND", "mock")

import asyncio
import pytest
from httpx import AsyncClient, ASGITransport

from main import app
from storage.database import init_db, engine, Base


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def setup_db():
    await init_db()
    yield
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "WatchLog"


@pytest.mark.asyncio
async def test_list_cameras_empty(client):
    resp = await client.get("/cameras")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_add_camera(client):
    resp = await client.post("/cameras", json={
        "name": "테스트 카메라",
        "rtsp_url": "rtsp://192.168.1.100:554/stream",
        "location": "입구",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "테스트 카메라"
    assert "rtsp_url" not in data
    return data["id"]


@pytest.mark.asyncio
async def test_list_alerts_empty(client):
    resp = await client.get("/alerts")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_alert_feedback_not_found(client):
    resp = await client.post("/alerts/99999/feedback", json={"false_positive": True})
    assert resp.status_code == 404
