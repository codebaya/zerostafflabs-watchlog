"""SQLite async database — cameras and alerts."""
from datetime import datetime
from typing import AsyncGenerator, List, Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config import settings

engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128))
    rtsp_url: Mapped[str] = mapped_column(String(512), unique=True)
    location: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(Integer)
    camera_name: Mapped[str] = mapped_column(String(128))
    event_type: Mapped[str] = mapped_column(String(64))        # "person", "anomaly", "intrusion", …
    description: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    snapshot_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    telegram_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    false_positive: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session


# ---------------------------------------------------------------------------
# Repository helpers
# ---------------------------------------------------------------------------

async def list_cameras(db: AsyncSession) -> List[Camera]:
    result = await db.execute(select(Camera))
    return list(result.scalars().all())


async def get_camera(db: AsyncSession, camera_id: int) -> Optional[Camera]:
    return await db.get(Camera, camera_id)


async def create_camera(db: AsyncSession, name: str, rtsp_url: str, location: Optional[str] = None) -> Camera:
    cam = Camera(name=name, rtsp_url=rtsp_url, location=location)
    db.add(cam)
    await db.commit()
    await db.refresh(cam)
    return cam


async def list_alerts(
    db: AsyncSession,
    camera_id: Optional[int] = None,
    limit: int = 50,
) -> List[Alert]:
    q = select(Alert).order_by(Alert.detected_at.desc()).limit(limit)
    if camera_id is not None:
        q = q.where(Alert.camera_id == camera_id)
    result = await db.execute(q)
    return list(result.scalars().all())


async def create_alert(
    db: AsyncSession,
    camera_id: int,
    camera_name: str,
    event_type: str,
    description: str,
    confidence: float,
    snapshot_path: Optional[str] = None,
) -> Alert:
    alert = Alert(
        camera_id=camera_id,
        camera_name=camera_name,
        event_type=event_type,
        description=description,
        confidence=confidence,
        snapshot_path=snapshot_path,
    )
    db.add(alert)
    await db.commit()
    await db.refresh(alert)
    return alert


async def mark_alert_sent(db: AsyncSession, alert_id: int) -> None:
    alert = await db.get(Alert, alert_id)
    if alert:
        alert.telegram_sent = True
        await db.commit()


async def mark_alert_feedback(db: AsyncSession, alert_id: int, false_positive: bool) -> Optional[Alert]:
    alert = await db.get(Alert, alert_id)
    if alert:
        alert.false_positive = false_positive
        await db.commit()
        await db.refresh(alert)
    return alert
