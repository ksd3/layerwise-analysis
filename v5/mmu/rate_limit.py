"""Minimal in-process token broker for rate-limited service access."""
from __future__ import annotations

from contextlib import contextmanager
from threading import Condition
from typing import Iterator


class TokenBroker:
    """A deterministic token broker used by workers or tests.

    The full cluster design can front this with a single broker process. This
    object provides the tested accounting semantics: no more than ``capacity``
    active acquisitions and every acquisition must be released.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._available = capacity
        self._cond = Condition()

    @property
    def available(self) -> int:
        return self._available

    @property
    def in_use(self) -> int:
        return self.capacity - self._available

    def acquire(self) -> None:
        with self._cond:
            while self._available <= 0:
                self._cond.wait()
            self._available -= 1

    def release(self) -> None:
        with self._cond:
            if self._available >= self.capacity:
                raise RuntimeError("release called without a matching acquire")
            self._available += 1
            self._cond.notify()

    @contextmanager
    def token(self) -> Iterator[None]:
        self.acquire()
        try:
            yield
        finally:
            self.release()
