"""
Central configuration loaded from environment variables / .env file.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    APP_NAME: str = "ReHub Health Monitor API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = (
        "postgresql+asyncpg://rehub:rehub_secret@localhost:5432/rehub"
    )
    DB_ECHO: bool = False

    # ML
    WINDOW_SIZE: int = 50           # readings per analysis window
    WINDOW_STEP: int = 10           # step between windows (overlap)
    MODEL_DIR: str = "app/ml/models"

    # Alert thresholds
    FATIGUE_THRESHOLD: float = 0.65
    STRAIN_THRESHOLD: float = 0.60
    INJURY_THRESHOLD: float = 0.55
    OVEREXERTION_THRESHOLD: float = 0.70
    TEMP_HIGH_THRESHOLD: float = 38.5   # °C skin temperature

    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 30    # seconds

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
