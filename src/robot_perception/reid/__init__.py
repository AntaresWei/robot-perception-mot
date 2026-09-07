"""Optional appearance encoding and identity-gallery utilities."""

from robot_perception.reid.fastreid import FastReIDEncoder
from robot_perception.reid.gallery import AppearanceGallery, GalleryMatch
from robot_perception.reid.manager import ReIDIdentityManager, ReIDMatchDecision

__all__ = [
    "AppearanceGallery",
    "FastReIDEncoder",
    "GalleryMatch",
    "ReIDIdentityManager",
    "ReIDMatchDecision",
]
