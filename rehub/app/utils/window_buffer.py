"""
In-memory sliding window buffer per device.

Each device maintains a deque of the last N SensorReading objects.
When the deque reaches WINDOW_SIZE, an analysis is triggered.
After analysis the deque slides forward by WINDOW_STEP readings.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Deque, Dict, List, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.database.models import SensorReading

logger = get_logger(__name__)


class SlidingWindowBuffer:
    """Thread-safe (asyncio) per-device sliding window."""

    def __init__(
        self,
        window_size: int = settings.WINDOW_SIZE,
        step: int = settings.WINDOW_STEP,
    ) -> None:
        self._window_size = window_size
        self._step = step
        # device_id → deque of SensorReading
        self._buffers: Dict[str, Deque[SensorReading]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _ensure_device(self, device_id: str) -> None:
        if device_id not in self._buffers:
            self._buffers[device_id] = deque(maxlen=self._window_size)
            self._locks[device_id] = asyncio.Lock()

    async def add(
        self, device_id: str, reading: SensorReading
    ) -> Optional[List[SensorReading]]:
        """
        Add a reading to the device's buffer.

        Returns a snapshot (list) of WINDOW_SIZE readings when a window is
        complete and analysis should run, otherwise returns None.
        """
        self._ensure_device(device_id)
        async with self._locks[device_id]:
            buf = self._buffers[device_id]
            buf.append(reading)

            if len(buf) >= self._window_size:
                window = list(buf)
                # Slide: remove oldest STEP readings
                for _ in range(self._step):
                    if buf:
                        buf.popleft()
                logger.debug(
                    "Window ready for device=%s size=%d remaining=%d",
                    device_id, len(window), len(buf),
                )
                return window

        return None

    async def buffer_size(self, device_id: str) -> int:
        self._ensure_device(device_id)
        return len(self._buffers[device_id])

    async def clear(self, device_id: str) -> None:
        self._ensure_device(device_id)
        async with self._locks[device_id]:
            self._buffers[device_id].clear()


# Module-level singleton shared across the application
window_buffer = SlidingWindowBuffer()
