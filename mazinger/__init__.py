"""
Mazinger Dubber -- End-to-end video dubbing pipeline.

Transcribe, translate, and voice-clone audio from any video URL.
Each stage can be used independently or chained through the unified
``MazingerDubber`` pipeline class.
"""

from mazinger.pipeline import MazingerDubber
from mazinger.paths import ProjectPaths
from mazinger.utils import LLMUsageTracker

__all__ = ["MazingerDubber", "ProjectPaths", "LLMUsageTracker"]
__version__ = "1.1.0"
