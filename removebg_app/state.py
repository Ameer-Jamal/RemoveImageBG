"""Domain models and state management utilities."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image


@dataclass
class ImageState:
    """Mutable editing parameters for a processed image."""

    hue_shift: int = 0
    contrast: float = 1.0
    brightness: float = 1.0
    blacks: float = 1.0
    whites: float = 1.0
    edge_smooth: int = 3
    background_mode: str = "transparent"
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    gradient_start: Tuple[int, int, int, int] = (255, 255, 255, 255)
    gradient_end: Tuple[int, int, int, int] = (0, 0, 0, 255)
    background_image_path: Optional[str] = None
    rotation: int = 0
    crop_box: Optional[Tuple[int, int, int, int]] = None
    target_size: Optional[Tuple[int, int]] = None

    def copy(self) -> "ImageState":
        """Return a deep copy of the current state."""

        return replace(self)


class HistoryManager:
    """Undo/redo stack for :class:`ImageState` instances."""

    def __init__(self) -> None:
        self._entries: List[ImageState] = []
        self._index: int = -1

    def clear(self) -> None:
        self._entries.clear()
        self._index = -1

    def push(self, state: ImageState) -> None:
        """Record a snapshot of ``state`` as the latest history entry."""

        snapshot = state.copy()
        if self._index + 1 < len(self._entries):
            self._entries = self._entries[: self._index + 1]
        self._entries.append(snapshot)
        self._index += 1

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return self._index + 1 < len(self._entries)

    def undo(self) -> Optional[ImageState]:
        if not self.can_undo:
            return None
        self._index -= 1
        return self._entries[self._index].copy()

    def redo(self) -> Optional[ImageState]:
        if not self.can_redo:
            return None
        self._index += 1
        return self._entries[self._index].copy()

    def current(self) -> Optional[ImageState]:
        if 0 <= self._index < len(self._entries):
            return self._entries[self._index].copy()
        return None


@dataclass
class ProcessedImage:
    """Model representing an input image and its derived state."""

    source_path: Path
    background_free: Image.Image
    state: ImageState = field(default_factory=ImageState)
    history: HistoryManager = field(default_factory=HistoryManager)

    @property
    def display_name(self) -> str:
        return self.source_path.name

    def snapshot(self) -> None:
        """Record the current state in the history manager."""

        self.history.push(self.state)
