"""SQLite history stores summaries, never uploaded rows."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def connect(path='data/modelwatch.sqlite3'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, summary TEXT NOT NULL)')
    return db


def save_run(result, path='data/modelwatch.sqlite3'):
    with connect(path) as db:
        cursor = db.execute('INSERT INTO runs (created_at, summary) VALUES (?, ?)', (datetime.now(timezone.utc).isoformat(), json.dumps(result, allow_nan=False)))
        return cursor.lastrowid


def history(path='data/modelwatch.sqlite3'):
    with connect(path) as db:
        return [(i, date, json.loads(payload)) for i, date, payload in db.execute('SELECT id, created_at, summary FROM runs ORDER BY id DESC')]


def clear(path='data/modelwatch.sqlite3'):
    with connect(path) as db:
        db.execute('DELETE FROM runs')
