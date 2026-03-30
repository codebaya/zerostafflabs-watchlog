from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # AI backends
    qwen_backend: str = "mock"       # "ollama" | "transformers" | "mock"
    qwen_model: str = "qwen2.5vl:7b"
    yolo_model_path: str = ""        # empty = auto-download latest YOLO
    yolo_confidence: float = 0.5

    # Storage
    database_url: str = "sqlite+aiosqlite:///./watchlog.db"
    clip_storage_dir: str = "./clips"

    # Night batch
    batch_start_hour: int = 2
    batch_end_hour: int = 6

    # Morning report
    report_hour: int = 8
    report_minute: int = 0

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


settings = Settings()
