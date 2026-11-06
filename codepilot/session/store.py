"""A trajectory's state and append-only evidence in one local SQLite database."""

import fcntl
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any


class SessionStore:
    def __init__(self, directory: Path, *, exclusive=True):
        directory.mkdir(parents=True, exist_ok=True)
        self.lease = (directory / "session.lock").open("a") if exclusive else None
        if self.lease is not None:
            try:
                fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.lease.close()
                raise ValueError("Session already open in another terminal") from None
        self.lock = RLock()
        self.db = sqlite3.connect(directory / "state.sqlite", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                kind TEXT NOT NULL, payload TEXT NOT NULL
            );
        """)

    def load(self):
        with self.lock:
            row = self.db.execute("SELECT value FROM state WHERE id=1").fetchone()
            return json.loads(row[0]) if row else None

    def save(self, state):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO state VALUES (1, ?)", (json.dumps(state),))

    def append(self, kind, payload, state=None):
        with self.lock, self.db:
            cursor = self.db.execute(
                "INSERT INTO events(timestamp,kind,payload) VALUES (?,?,?)",
                (datetime.now(timezone.utc).isoformat(), kind, json.dumps(payload, default=str)),
            )
            if state is not None:
                self.db.execute("INSERT OR REPLACE INTO state VALUES (1, ?)", (json.dumps(state),))
            return cursor.lastrowid

    def event_info(self, ids):
        with self.lock:
            return [
                {"id": event_id, "timestamp": timestamp, "kind": kind}
                for event_id, timestamp, kind in self.db.execute(
                    "SELECT id,timestamp,kind FROM events WHERE id IN ("
                    + ",".join("?" for _ in ids)
                    + ")",
                    list(ids),
                ).fetchall()
            ]

    def query(
        self, *, event_id=None, keyword="", offset=0, limit=20, max_chars=12000, content_offset=0
    ):
        if (
            content_offset < 0
            or (content_offset and event_id is None)
            or offset < 0
            or not 1 <= limit <= 100
            or not 500 <= max_chars <= 32000
        ):
            raise ValueError("Invalid archive page bounds")
        where = "WHERE id=?" if event_id is not None else "WHERE instr(payload, ?) > 0"
        value = event_id if event_id is not None else keyword
        with self.lock:
            rows = self.db.execute(
                f"SELECT id,timestamp,kind,payload FROM events {where} ORDER BY id LIMIT ? OFFSET ?",
                (value, limit + 1, offset),
            ).fetchall()
        events: list[dict[str, Any]] = []
        remaining = max_chars - 200
        consumed = 0
        for event, timestamp, kind, payload in rows[:limit]:
            entry = {"id": event, "timestamp": timestamp, "kind": kind, "payload": payload}
            if content_offset:
                entry["payload"] = payload[content_offset:]
                entry["content_offset"] = content_offset
            size = len(json.dumps(entry, ensure_ascii=False))
            if size > remaining:
                if events:
                    break
                # Keep a handle to the full event, with explicit truncation metadata.
                entry["content_offset"] = content_offset
                entry["truncated"] = True
                low, high = 0, len(payload) - content_offset
                while low < high:
                    middle = (low + high + 1) // 2
                    entry["payload"] = payload[content_offset : content_offset + middle]
                    entry["next_content_offset"] = content_offset + middle
                    if len(json.dumps(entry, ensure_ascii=False)) <= remaining:
                        low = middle
                    else:
                        high = middle - 1
                entry["payload"] = payload[content_offset : content_offset + low]
                entry["next_content_offset"] = content_offset + low
            events.append(entry)
            consumed += 1
            remaining -= len(json.dumps(entry, ensure_ascii=False))
        return {
            "events": events,
            "offset": offset,
            "next_offset": offset + consumed if len(rows) > consumed else None,
        }

    def close(self):
        try:
            self.db.close()
        finally:
            if self.lease is not None:
                self.lease.close()
