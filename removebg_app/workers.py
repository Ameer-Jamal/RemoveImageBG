"""Background processing helpers leveraging multiple CPU cores."""

from __future__ import annotations

import atexit
import io
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Iterable, List, Optional

from PyQt5.QtCore import QObject, QThread, pyqtSignal


_SESSION = None
_EXECUTOR: Optional[ProcessPoolExecutor] = None
_EXECUTOR_SESSION: Optional[str] = None
_EXECUTOR_WORKERS: Optional[int] = None
_EXECUTOR_LOCK = threading.Lock()


def _initialise_session(session_name: str) -> None:
    """Initialise a ``rembg`` session for each worker process."""

    global _SESSION
    from rembg import new_session

    _SESSION = new_session(session_name)


def _process_path(path: str) -> bytes:
    """Worker entry point returning a PNG-encoded RGBA image."""

    if _SESSION is None:  # pragma: no cover - defensive guard
        raise RuntimeError("rembg session was not initialised in worker")

    from PIL import Image

    image_path = Path(path)
    with Image.open(image_path) as loaded:
        image = loaded.convert("RGBA")

    from rembg import remove

    result = remove(image, session=_SESSION)
    image.close()

    from removebg_app.effects import refine_alpha

    refined = refine_alpha(result, radius=3)
    result.close()
    with io.BytesIO() as buffer:
        refined.save(buffer, format="PNG")
        return buffer.getvalue()


def _get_executor(session_name: str, max_workers: int) -> ProcessPoolExecutor:
    """Return a shared executor, reusing warm worker processes when possible."""

    global _EXECUTOR, _EXECUTOR_SESSION, _EXECUTOR_WORKERS

    with _EXECUTOR_LOCK:
        if (
            _EXECUTOR is None
            or _EXECUTOR_SESSION != session_name
            or _EXECUTOR_WORKERS != max_workers
        ):
            if _EXECUTOR is not None:
                _EXECUTOR.shutdown(wait=False, cancel_futures=True)
            _EXECUTOR = ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_initialise_session,
                initargs=(session_name,),
            )
            _EXECUTOR_SESSION = session_name
            _EXECUTOR_WORKERS = max_workers
    return _EXECUTOR


def _shutdown_executor() -> None:
    global _EXECUTOR, _EXECUTOR_SESSION, _EXECUTOR_WORKERS
    with _EXECUTOR_LOCK:
        if _EXECUTOR is not None:
            _EXECUTOR.shutdown(wait=False, cancel_futures=True)
        _EXECUTOR = None
        _EXECUTOR_SESSION = None
        _EXECUTOR_WORKERS = None


atexit.register(_shutdown_executor)


def warm_process_pool(session_name: str, *, max_workers: Optional[int] = None) -> None:
    """Ensure the shared executor is created so the first batch is responsive."""

    workers = max_workers or BackgroundRemovalWorker.recommended_worker_count()
    _get_executor(session_name, workers)


class BackgroundRemovalWorker(QThread):
    """Dispatch heavy background removal work to a process pool."""

    item_ready = pyqtSignal(str, bytes)
    progress = pyqtSignal(int, int)
    failed = pyqtSignal(str, str)

    def __init__(
        self,
        paths: Iterable[Path],
        *,
        session_name: str,
        max_workers: Optional[int] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._paths: List[Path] = list(paths)
        self._session_name = session_name
        self._max_workers = max_workers or self.recommended_worker_count()

    @staticmethod
    def recommended_worker_count() -> int:
        detected = cpu_count() or 1
        # Keep at least one free core for the UI/main thread.
        return max(1, min(detected - 1, 4))

    def run(self) -> None:  # pragma: no cover - exercised via UI
        if not self._paths:
            return

        total = len(self._paths)
        completed = 0

        executor = _get_executor(self._session_name, self._max_workers)
        futures = {executor.submit(_process_path, str(path)): path for path in self._paths}

        for future in as_completed(futures):
            path = futures[future]
            try:
                payload = future.result()
            except Exception as exc:  # pragma: no cover - runtime safeguard
                self.failed.emit(str(path), str(exc))
            else:
                self.item_ready.emit(str(path), payload)
            finally:
                completed += 1
                self.progress.emit(completed, total)
