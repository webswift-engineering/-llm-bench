"""Pricing snapshot save/load."""

from __future__ import annotations

import json
from pathlib import Path

from llm_bench.models import PricingSnapshot, utc_now
from llm_bench.pricing.catalog import get_catalog

SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "data" / "snapshots"


def save_snapshot() -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = PricingSnapshot(captured_at=utc_now(), models=get_catalog())
    filename = snapshot.captured_at.strftime("%Y-%m-%d.json")
    path = SNAPSHOT_DIR / filename
    path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
    return path


def load_latest_snapshot() -> PricingSnapshot | None:
    if not SNAPSHOT_DIR.exists():
        return None
    files = sorted(SNAPSHOT_DIR.glob("*.json"), reverse=True)
    if not files:
        return None
    data = json.loads(files[0].read_text(encoding="utf-8"))
    return PricingSnapshot.from_dict(data)
