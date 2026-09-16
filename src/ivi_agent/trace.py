"""Structured run trace: machine-readable events + a human-readable log.

Every run emits, alongside the existing ``result.json`` / ``report.html``:

* ``events.jsonl`` — one JSON object per line, the ordered event stream of the
  run (plan, retrieval, decision, execute, verify, incident, vehicle_state,
  done). This is the backbone for replay, diagnostics, and an engineer report.
* ``agent.log`` — the same events rendered as timestamped text lines.
* ``task.json`` — the goal and a config snapshot captured at start.
* ``plan.json`` — the planned subgoals.

The event schema is deliberately open (``event(kind, **fields)``) so new signal
sources — VHAL/CAN vehicle state, QNX compositor, system logs — can be appended
as their own event kinds without changing this module. Everything is best-effort:
a tracing failure never aborts a run.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunTrace:
    """Append-only event + log writer for a single run directory."""

    def __init__(self, directory: Path, run_id: str, enabled: bool = True) -> None:
        self.directory = directory
        self.run_id = run_id
        self.enabled = enabled
        self._events_path = directory / "events.jsonl"
        self._seq = 0
        self._logger: logging.Logger | None = None
        if not enabled:
            return
        try:
            self._events_path.touch()
            logger = logging.getLogger(f"ivi_agent.run.{run_id}")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            # Avoid duplicate handlers if a run id ever repeats in-process.
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
            file_handler = logging.FileHandler(directory / "agent.log", encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            )
            logger.addHandler(file_handler)
            self._logger = logger
        except Exception:  # noqa: BLE001 - tracing must never break a run
            self.enabled = False

    def event(self, kind: str, message: str | None = None, **fields: Any) -> None:
        """Record one event to events.jsonl and agent.log.

        ``kind`` is a stable category (e.g. "plan", "decision", "execute",
        "verify", "incident", "vehicle_state", "done"). ``fields`` are arbitrary
        JSON-serializable details. ``message`` is an optional human summary for
        the text log; it defaults to the kind.
        """
        if not self.enabled:
            return
        self._seq += 1
        record = {
            "seq": self._seq,
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            **fields,
        }
        try:
            with self._events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:  # noqa: BLE001
            pass
        if self._logger is not None:
            summary = message or kind
            detail = " ".join(f"{key}={value!r}" for key, value in fields.items())
            try:
                self._logger.info("[%s] %s %s", kind, summary, detail)
            except Exception:  # noqa: BLE001
                pass

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        """Write a standalone JSON artifact (task.json, plan.json, …)."""
        if not self.enabled:
            return
        try:
            (self.directory / name).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        if self._logger is not None:
            for handler in list(self._logger.handlers):
                try:
                    handler.close()
                except Exception:  # noqa: BLE001
                    pass
                self._logger.removeHandler(handler)
            self._logger = None


class NullTrace(RunTrace):
    """A trace that writes nothing — used when tracing is disabled."""

    def __init__(self) -> None:  # noqa: D401 - see base
        self.enabled = False
        self.directory = Path(".")
        self.run_id = ""
        self._logger = None

    def event(self, *args: Any, **kwargs: Any) -> None:
        return

    def write_json(self, *args: Any, **kwargs: Any) -> None:
        return

    def close(self) -> None:
        return
