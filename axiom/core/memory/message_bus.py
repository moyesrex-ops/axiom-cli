"""Async pub/sub message bus for cross-channel notifications.

When a message arrives on Telegram, CLI subscribers get notified
(and vice versa), enabling the mirrored conversation experience.
"""

import asyncio
import logging
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


class MessageBus:
    """Lightweight async pub/sub bus.

    Subscribers register callbacks for specific channels.
    When a message is published, all subscribers on OTHER channels
    get notified (source channel is excluded to prevent echo).
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable[..., Awaitable]]] = {}

    def subscribe(
        self, channel: str, callback: Callable[..., Awaitable]
    ) -> None:
        """Subscribe to messages on a channel.

        Args:
            channel: Channel name ("cli", "telegram", "*" for all)
            callback: async def callback(message: dict) -> None
        """
        self._subscribers.setdefault(channel, []).append(callback)

    def unsubscribe(
        self, channel: str, callback: Callable[..., Awaitable]
    ) -> None:
        """Remove a subscription."""
        if channel in self._subscribers:
            self._subscribers[channel] = [
                cb for cb in self._subscribers[channel] if cb is not callback
            ]

    async def publish(self, message: dict, source_channel: str) -> None:
        """Publish a message, notifying subscribers on other channels.

        Args:
            message: The message dict (role, content, channel, ts, metadata)
            source_channel: Where the message originated (don't echo back)
        """
        tasks = []
        for channel, callbacks in self._subscribers.items():
            # Don't echo back to source channel, but "*" always gets notified
            if channel == source_channel and channel != "*":
                continue
            for cb in callbacks:
                tasks.append(asyncio.create_task(self._safe_call(cb, message)))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def _safe_call(cb: Callable, message: dict) -> None:
        """Call callback with error suppression — one bad subscriber
        must not break others."""
        try:
            await cb(message)
        except Exception as exc:
            logger.debug("MessageBus callback error: %s", exc)
