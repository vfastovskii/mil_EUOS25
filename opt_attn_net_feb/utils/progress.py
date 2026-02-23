from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator
import traceback
import time


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _fmt_kv(**kwargs: Any) -> str:
    parts: list[str] = []
    for k, v in kwargs.items():
        if v is None:
            continue
        try:
            s = str(v)
        except Exception:
            s = repr(v)
        parts.append(f"{k}={s}")
    return " ".join(parts)


def log_event(event: str, step: str, **kwargs: Any) -> None:
    """
    Prints a structured pipeline progress message with timestamp.

    Output format:
      [YYYY-mm-dd HH:MM:SS] [EVENT] step key=value ...
    """
    kv = _fmt_kv(**kwargs)
    suffix = f" {kv}" if kv else ""
    print(f"[{_now_ts()}] [{event}] {step}{suffix}", flush=True)


@contextmanager
def log_step(step: str, **kwargs: Any) -> Iterator[None]:
    """
    Context manager for START/DONE/FAIL logging with elapsed seconds.
    """
    t0 = time.perf_counter()
    log_event("START", step, **kwargs)
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        log_event("FAIL", step, elapsed_s=f"{elapsed:.2f}", error=repr(exc))
        traceback.print_exc()
        raise
    elapsed = time.perf_counter() - t0
    log_event("DONE", step, elapsed_s=f"{elapsed:.2f}")


__all__ = ["log_event", "log_step"]
