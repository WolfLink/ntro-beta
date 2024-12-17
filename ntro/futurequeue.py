"""A module for a queue of futures that allows for asynchronous iteration."""
from __future__ import annotations

from typing import Any

from bqskit.runtime import get_runtime


class FutureQueue:
    def __init__(self, future: Any, length: int) -> None:
        self.future: Any = future
        self.queue: list[tuple[int, Any]] = []
        self.remaining: int = length

    def __aiter__(self) -> FutureQueue:
        return self

    async def __anext__(self) -> Any:
        if len(self.queue) > 0:
            self.remaining -= 1
            return self.queue.pop(0)

        elif self.remaining < 1:
            raise StopAsyncIteration

        else:
            try:
                value = await get_runtime().next(self.future)
                self.queue.extend(value)
                return self.queue.pop(0)

            except IndexError:
                raise StopAsyncIteration
