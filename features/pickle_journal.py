"""Crash-recoverable append-only pickle journals for long feature jobs."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Iterator


_TRUNCATED_PICKLE_ERRORS = (
    EOFError,
    pickle.UnpicklingError,
    AttributeError,
    ImportError,
    IndexError,
    ValueError,
)


def iter_pickle_records(
    path: str | os.PathLike[str],
    *,
    repair_trailing_record: bool = False,
    ignore_trailing_record: bool = False,
) -> Iterator[Any]:
    """Yield legacy list-shard or append-stream records from ``path``.

    New per-card journals contain one independently pickled record after
    another. If a worker stops during its final append, resume truncates only
    that incomplete trailing record.
    """

    source = Path(path)
    mode = "r+b" if repair_trailing_record else "rb"
    with source.open(mode) as handle:
        size = source.stat().st_size
        while handle.tell() < size:
            record_start = handle.tell()
            try:
                value = pickle.load(handle)
            except _TRUNCATED_PICKLE_ERRORS as exc:
                if ignore_trailing_record:
                    break
                if not repair_trailing_record:
                    raise ValueError(
                        f"Invalid pickle journal record at byte {record_start}: {source}"
                    ) from exc
                handle.seek(record_start)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
                break
            if isinstance(value, list):
                # Backwards compatibility with the former atomic list shards.
                yield from value
            else:
                yield value


class AppendPickleJournal:
    """One append-only pickle stream, normally one file per GPU worker."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        resume: bool = True,
        sync_every: int = 32,
        recover: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sync_every = max(1, int(sync_every))
        if self.path.exists():
            if not resume:
                raise FileExistsError(f"Pickle journal already exists: {self.path}")
            if recover:
                for _ in iter_pickle_records(
                    self.path, repair_trailing_record=True
                ):
                    pass
        self._handle = None
        self._since_sync = 0

    def append(self, value: Any) -> None:
        if self._handle is None:
            # Unbuffered writes expose every completed record immediately.
            self._handle = self.path.open("ab", buffering=0)
        pickle.dump(value, self._handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._since_sync += 1
        if self._since_sync >= self.sync_every:
            self.flush()

    def flush(self) -> None:
        if self._handle is None:
            return
        os.fsync(self._handle.fileno())
        self._since_sync = 0

    def close(self) -> None:
        if self._handle is None:
            return
        self.flush()
        self._handle.close()
        self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
