"""Local raw archive, canonical Parquet partitions, manifest and checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .core import TRADE_COLUMNS, validate_trades


def _json_safe(value: Any) -> Any:
    """Make exchange responses (and their odd values) JSON-serializable.

    Huge integers cannot be converted to text in CPython (int-to-str digit
    limit) and NaN/Inf are not valid JSON, so both are replaced with markers
    instead of aborting the whole raw-page archive.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value if value.bit_length() <= 63 else f"<int:bits={value.bit_length()}>"
    if isinstance(value, float):
        return value if math.isfinite(value) else f"<float:{value!r}>"
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        return str(value)
    except (ValueError, TypeError, OverflowError):
        return f"<{type(value).__name__}>"


class TradeStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.raw = self.root / "raw"
        self.canonical = self.root / "canonical"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.canonical.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.checkpoint_path = self.root / "checkpoint.json"

    # ------------------------------------------------------------------ #
    # Raw archive
    # ------------------------------------------------------------------ #
    def save_raw(self, exchange: str, symbol: str, since_ms: int, rows: list[Any]) -> Path:
        path = self.raw / exchange / _encode(symbol)
        path.mkdir(parents=True, exist_ok=True)
        suffix = len(list(path.glob(f"{since_ms}-*.jsonl")))
        file = path / f"{since_ms}-{suffix:06d}.jsonl"
        with file.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(_json_safe(row), separators=(",", ":")) + "\n"
                )
        return file

    # ------------------------------------------------------------------ #
    # Canonical Parquet partitions
    # ------------------------------------------------------------------ #
    def write(self, trades: pd.DataFrame, metadata: dict[str, Any] | None = None) -> Path:
        frame, report = validate_trades(trades)
        if frame.empty:
            raise ValueError("No valid trades to persist")
        frame = frame.assign(_date=frame["timestamp"].dt.strftime("%Y-%m-%d"))
        written = []
        for (exchange, symbol_value, date), part in frame.groupby(
            ["exchange", "symbol", "_date"], sort=True
        ):
            symbol = _encode(str(symbol_value))
            part = part.drop(columns=["_date"])
            folder = self.canonical / f"exchange={exchange}" / f"symbol={symbol}" / f"date={date}"
            folder.mkdir(parents=True, exist_ok=True)
            file = folder / "trades.parquet"
            if file.exists():
                existing = pd.read_parquet(file)
                part = pd.concat([existing, part], ignore_index=True)
                part, _ = validate_trades(part)
            # Write-then-replace: a crash mid-write must not corrupt the
            # partition that already holds valid history.
            temp = file.with_name(file.name + ".tmp")
            part.to_parquet(temp, index=False)
            temp.replace(file)
            written.append(str(file.relative_to(self.root)))
        manifest = self._read_json(self.manifest_path, {"schema_version": 1, "files": []})
        manifest["files"] = sorted(set(manifest["files"]) | set(written))
        manifest.setdefault("metadata", {}).update(metadata or {})
        manifest["last_validation"] = {
            "rows_input": report.rows_input,
            "rows_valid": report.rows_valid,
            "duplicates_removed": report.duplicates_removed,
            "invalid_rows_removed": report.invalid_rows_removed,
        }
        self._write_json(self.manifest_path, manifest)
        return self.canonical

    def read(
        self,
        exchange: str | None = None,
        symbol: str | None = None,
        *,
        sort: bool = False,
    ) -> pd.DataFrame:
        files = sorted(self.canonical.glob("exchange=*/symbol=*/date=*/trades.parquet"))
        if exchange:
            files = [p for p in files if p.parent.parent.parent.name == f"exchange={exchange}"]
        if symbol:
            encoded = _encode(symbol)
            files = [p for p in files if p.parent.parent.name == f"symbol={encoded}"]
        if not files:
            return pd.DataFrame(columns=TRADE_COLUMNS)
        frame = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
        if sort:
            frame = frame.sort_values(
                ["timestamp", "exchange", "trade_id"], kind="stable",
                na_position="last",
            ).reset_index(drop=True)
        return frame

    # ------------------------------------------------------------------ #
    # Checkpoint and manifest notes
    # ------------------------------------------------------------------ #
    def set_checkpoint(self, exchange: str, symbol: str, next_since_ms: int) -> None:
        data = self._read_json(self.checkpoint_path, {})
        data[f"{exchange}:{symbol}"] = int(next_since_ms)
        self._write_json(self.checkpoint_path, data)

    def get_checkpoint(self, exchange: str, symbol: str) -> int | None:
        return self._read_json(self.checkpoint_path, {}).get(f"{exchange}:{symbol}")

    def note_issue(self, exchange: str, symbol: str, message: str) -> None:
        """Record a data-quality warning so it survives the console."""
        manifest = self._read_json(self.manifest_path, {"schema_version": 1, "files": []})
        issues = manifest.setdefault("issues", [])
        issues.append({
            "exchange": exchange,
            "symbol": symbol,
            "message": message,
            "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        if len(issues) > 200:
            del issues[:-200]
        self._write_json(self.manifest_path, manifest)

    def record_download(self, exchange: str, symbol: str, report: Any) -> None:
        """Persist the summary of a download run into the manifest."""
        manifest = self.manifest_data()
        contract_size = getattr(report, "contract_size", None)
        manifest.setdefault("downloads", []).append({
            "exchange": exchange,
            "symbol": symbol,
            "pages": getattr(report, "pages", 0),
            "rows_fetched": getattr(report, "rows_fetched", 0),
            "rows_written": getattr(report, "rows_written", 0),
            "malformed_rows": getattr(report, "malformed_rows", 0),
            "contract_size": contract_size,
            "incomplete_buckets": list(getattr(report, "incomplete_buckets", [])),
            "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        if len(manifest["downloads"]) > 50:
            del manifest["downloads"][:-50]
        self._write_json(self.manifest_path, manifest)

    # ------------------------------------------------------------------ #
    # Manifest inspection
    # ------------------------------------------------------------------ #
    def manifest_data(self) -> dict[str, Any]:
        return self._read_json(self.manifest_path, {"schema_version": 1, "files": []})

    def checkpoints(self) -> dict[str, int]:
        return self._read_json(self.checkpoint_path, {})

    def last_contract_size(self, exchange: str, symbol: str) -> float | None:
        """Contract size used by the most recent download of this instrument."""
        for entry in reversed(self.manifest_data().get("downloads", [])):
            if entry.get("exchange") == exchange and entry.get("symbol") == symbol:
                size = entry.get("contract_size")
                try:
                    return None if size is None else float(size)
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(path)


def _encode(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")