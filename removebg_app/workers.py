"""Background processing helpers leveraging multiple CPU cores."""

from __future__ import annotations

import io
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Iterable, List, Optional

from PyQt5.QtCore import QObject, QThread, pyqtSignal


_SESSION = None


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
        if max_workers:
            self._max_workers = max_workers
        else:
            detected = cpu_count() or 1
            self._max_workers = max(1, detected - 1)

    def run(self) -> None:  # pragma: no cover - exercised via UI
        if not self._paths:
            return

        total = len(self._paths)
        completed = 0

        with ProcessPoolExecutor(
            max_workers=self._max_workers,
            initializer=_initialise_session,
            initargs=(self._session_name,),
        ) as executor:
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
