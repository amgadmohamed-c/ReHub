"""
ReHub FastAPI application factory.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.database.session import init_db
from app.ml.model_registry import ModelRegistry
from app.routes import (
    alert_routes,
    analysis_routes,
    health_routes,
    sensor_routes,
    ws_routes,
    simulate_routes,
)

setup_logging()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    # Startup
    logger.info("Starting ReHub backend v%s", settings.VERSION)
    await init_db()
    ModelRegistry.load()
    logger.info("ML models loaded. Ready to serve.")
    yield
    # Shutdown
    logger.info("Shutting down ReHub backend.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=(
        "AI-powered wearable health monitoring backend. "
        "Accepts sensor payloads from Arduino wearables via a Flutter/React Native "
        "mobile app, runs ML exercise classification and injury prediction on "
        "sliding windows of data, and streams real-time alerts via WebSocket."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS (allow all origins for mobile app development)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health_routes.router)
app.include_router(sensor_routes.router, prefix="/api/v1")
app.include_router(analysis_routes.router, prefix="/api/v1")
app.include_router(alert_routes.router, prefix="/api/v1")
app.include_router(ws_routes.router)
app.include_router(simulate_routes.router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
    }