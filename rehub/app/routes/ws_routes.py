"""
WebSocket endpoint: ws://<host>/ws/{device_id}

Mobile apps connect here to receive real-time analysis results and alerts.
The connection is kept alive by periodic heartbeats.
"""

import asyncio
import json
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.logging import get_logger
from app.utils.ws_manager import ws_manager

router = APIRouter(tags=["WebSocket"])
logger = get_logger(__name__)


@router.websocket("/ws/{device_id}")
async def websocket_endpoint(device_id: str, ws: WebSocket) -> None:
    """
    Real-time channel for a wearable device.

    Messages pushed by the server:
      • event="analysis"  – after every completed sliding window
      • event="alert"     – when a health threshold is breached
      • event="heartbeat" – every N seconds to keep the connection alive

    Messages the client can send:
      • {"event": "ping"}  – server replies with {"event": "pong"}
    """
    await ws_manager.connect(device_id, ws)
    logger.info("WS session started: device=%s", device_id)

    # Send welcome message
    await ws.send_text(json.dumps({
        "event": "connected",
        "device_id": device_id,
        "timestamp": datetime.utcnow().isoformat(),
        "message": "ReHub health monitor connected. Awaiting sensor data.",
    }))

    heartbeat_task = asyncio.create_task(_heartbeat(device_id, ws))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                if msg.get("event") == "ping":
                    await ws.send_text(json.dumps({
                        "event": "pong",
                        "timestamp": datetime.utcnow().isoformat(),
                    }))
            except json.JSONDecodeError:
                pass  # ignore malformed messages

    except WebSocketDisconnect:
        logger.info("WS disconnected: device=%s", device_id)
    finally:
        heartbeat_task.cancel()
        ws_manager.disconnect(device_id, ws)


async def _heartbeat(device_id: str, ws: WebSocket) -> None:
    """Send periodic heartbeats to keep the connection alive."""
    try:
        while True:
            await asyncio.sleep(settings.WS_HEARTBEAT_INTERVAL)
            await ws.send_text(json.dumps({
                "event": "heartbeat",
                "timestamp": datetime.utcnow().isoformat(),
            }))
    except Exception:
        pass  # socket closed; task will be cancelled by caller
