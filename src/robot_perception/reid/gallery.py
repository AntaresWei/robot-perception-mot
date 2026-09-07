"""Persistent appearance gallery for long-gap identity reactivation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GalleryMatch:
    track_id: int
    class_id: int
    similarity: float
    age_frames: int


@dataclass
class _GalleryEntry:
    track_id: int
    class_id: int
    embedding: np.ndarray
    last_seen_frame: int
    updates: int = 1
    last_score: float = 1.0


class AppearanceGallery:
    """Store EMA-smoothed embeddings independently of active Track objects."""

    def __init__(self, *, momentum: float = 0.9, max_age_frames: int = 300) -> None:
        if not 0.0 <= momentum < 1.0 or not np.isfinite(momentum):
            raise ValueError("momentum must be finite and in [0, 1)")
        if max_age_frames < 1:
            raise ValueError("max_age_frames must be positive")
        self.momentum = float(momentum)
        self.max_age_frames = int(max_age_frames)
        self._entries: dict[int, _GalleryEntry] = {}

    @staticmethod
    def _embedding(value: np.ndarray) -> np.ndarray:
        embedding = np.asarray(value, dtype=np.float32)
        if embedding.ndim != 1 or embedding.size == 0 or not np.isfinite(embedding).all():
            raise ValueError("embedding must be a finite non-empty vector")
        norm = float(np.linalg.norm(embedding))
        if norm <= np.finfo(np.float32).eps:
            raise ValueError("embedding must not be zero")
        return embedding / norm

    def update(self, track_id: int, class_id: int, embedding: np.ndarray, frame_id: int, *, score: float = 1.0) -> None:
        if track_id < 0 or class_id <= 0 or frame_id < 0:
            raise ValueError("track_id, class_id, and frame_id must be non-negative/positive")
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score must be finite and in [0, 1]")
        vector = self._embedding(embedding)
        entry = self._entries.get(int(track_id))
        if entry is None or entry.class_id != int(class_id) or entry.embedding.shape != vector.shape:
            self._entries[int(track_id)] = _GalleryEntry(int(track_id), int(class_id), vector.copy(), int(frame_id), 1, float(score))
            return
        blended = self.momentum * entry.embedding + (1.0 - self.momentum) * vector
        entry.embedding = self._embedding(blended)
        entry.last_seen_frame = int(frame_id)
        entry.updates += 1
        entry.last_score = float(score)

    def purge(self, frame_id: int) -> None:
        if frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        expired = [track_id for track_id, entry in self._entries.items() if frame_id - entry.last_seen_frame > self.max_age_frames]
        for track_id in expired:
            del self._entries[track_id]

    def match(
        self,
        embedding: np.ndarray,
        *,
        class_id: int,
        frame_id: int,
        min_similarity: float,
        min_gap_frames: int = 1,
        excluded_track_ids: set[int] | frozenset[int] = frozenset(),
    ) -> GalleryMatch | None:
        if not 0.0 <= min_similarity <= 1.0 or not np.isfinite(min_similarity):
            raise ValueError("min_similarity must be finite and in [0, 1]")
        if min_gap_frames < 1:
            raise ValueError("min_gap_frames must be positive")
        vector = self._embedding(embedding)
        self.purge(frame_id)
        candidates: list[GalleryMatch] = []
        for track_id, entry in self._entries.items():
            age = int(frame_id - entry.last_seen_frame)
            if entry.class_id != int(class_id) or track_id in excluded_track_ids or age < min_gap_frames:
                continue
            similarity = float(np.dot(vector, entry.embedding))
            if similarity >= min_similarity:
                candidates.append(GalleryMatch(track_id, entry.class_id, similarity, age))
        return max(candidates, key=lambda item: item.similarity) if candidates else None

    def __len__(self) -> int:
        return len(self._entries)
