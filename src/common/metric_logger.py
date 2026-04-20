from __future__ import annotations
import csv, json, time
from pathlib import Path
from typing import Any

class MetricLogger:
    def __init__(self, csv_path: str | Path, jsonl_path: str | Path, flush_every: int = 100):
        self.csv_path = Path(csv_path)
        self.jsonl_path = Path(jsonl_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._jsonl_file = self.jsonl_path.open('w', encoding='utf-8')
        self._rows: list[dict[str, Any]] = []
        self._fieldnames: list[str] = []
        self._flush_every = flush_every
        self._since_flush = 0

    def _rewrite_csv(self):
        with self.csv_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow({k: row.get(k, '') for k in self._fieldnames})

    def log(self, row: dict[str, Any]) -> None:
        if 'wall_time' not in row:
            row['wall_time'] = time.time()
        new_key = False
        for k in row.keys():
            if k not in self._fieldnames:
                self._fieldnames.append(k)
                new_key = True
        self._rows.append(dict(row))
        self._jsonl_file.write(json.dumps(row, ensure_ascii=False) + '\n')
        self._jsonl_file.flush()
        self._since_flush += 1
        if new_key or self._since_flush >= self._flush_every:
            self._rewrite_csv()
            self._since_flush = 0

    def flush(self) -> None:
        self._jsonl_file.flush()
        self._rewrite_csv()
        self._since_flush = 0

    def close(self) -> None:
        self.flush()
        self._jsonl_file.close()
