"""Gated identity recovery on top of a 3D Tracker's public IDs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_perception.reid.gallery import AppearanceGallery, GalleryMatch


@dataclass(frozen=True)
class ReIDMatchDecision:
    internal_track_id: int
    public_track_id: int
    similarity: float | None
    reactivated: bool


class ReIDIdentityManager:
    """Map internal 3D track IDs to stable public IDs using FastReID features.

    This wrapper does not mutate ``MultiObjectTracker``. It aliases a newly
    created internal track to an expired public identity only after motion/2D-3D
    association has supplied a reliable image crop. Thus Tracker remains fully
    ROS/Detector/ReID independent.
    """

    def __init__(
        self,
        encoder: object,
        *,
        gallery: AppearanceGallery | None = None,
        similarity_threshold: float = 0.75,
        min_gap_frames: int = 10,
        min_fusion_confidence: float = 0.5,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0 or not np.isfinite(similarity_threshold):
            raise ValueError("similarity_threshold must be finite and in [0, 1]")
        if min_gap_frames < 1:
            raise ValueError("min_gap_frames must be positive")
        if not 0.0 <= min_fusion_confidence <= 1.0 or not np.isfinite(min_fusion_confidence):
            raise ValueError("min_fusion_confidence must be finite and in [0, 1]")
        self.encoder = encoder
        self.gallery = gallery or AppearanceGallery()
        self.similarity_threshold = float(similarity_threshold)
        self.min_gap_frames = int(min_gap_frames)
        self.min_fusion_confidence = float(min_fusion_confidence)
        self._internal_to_public: dict[int, int] = {}

    def public_id(self, internal_track_id: int) -> int:
        return int(self._internal_to_public.get(int(internal_track_id), int(internal_track_id)))

    def resolve(
        self,
        *,
        internal_track_id: int,
        class_id: int,
        frame_id: int,
        frame: object,
        bbox: np.ndarray,
        fusion_confidence: float,
        active_public_ids: set[int] | frozenset[int],
        detection_score: float = 1.0,
    ) -> ReIDMatchDecision:
        """Update gallery for an existing track or attempt gated reactivation."""

        if not 0.0 <= fusion_confidence <= 1.0 or not np.isfinite(fusion_confidence):
            raise ValueError("fusion_confidence must be finite and in [0, 1]")
        embedding = self.encoder.encode(frame, bbox)
        if embedding is None or fusion_confidence < self.min_fusion_confidence:
            public = self.public_id(internal_track_id)
            return ReIDMatchDecision(int(internal_track_id), public, None, False)

        existing_public = self._internal_to_public.get(int(internal_track_id))
        if existing_public is not None:
            self.gallery.update(existing_public, class_id, embedding, frame_id, score=detection_score)
            return ReIDMatchDecision(int(internal_track_id), existing_public, None, False)

        match: GalleryMatch | None = self.gallery.match(
            embedding,
            class_id=class_id,
            frame_id=frame_id,
            min_similarity=self.similarity_threshold,
            min_gap_frames=self.min_gap_frames,
            excluded_track_ids=set(active_public_ids) | {int(internal_track_id)},
        )
        public = int(match.track_id) if match is not None else int(internal_track_id)
        self._internal_to_public[int(internal_track_id)] = public
        self.gallery.update(public, class_id, embedding, frame_id, score=detection_score)
        return ReIDMatchDecision(int(internal_track_id), public, None if match is None else match.similarity, match is not None)

    def remove_deleted(self, active_internal_ids: set[int] | frozenset[int]) -> None:
        """Drop aliases for deleted internal tracks; gallery remains dormant."""

        for internal_id in list(self._internal_to_public):
            if internal_id not in active_internal_ids:
                del self._internal_to_public[internal_id]
