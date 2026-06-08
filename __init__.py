"""
Dental Color Analyzer
Detect upper incisors and extract LAB colour values.
"""

from .analyzer import DentalColorAnalyzer
from .cli import main

__version__ = "2.0.0"
__all__ = ["DentalColorAnalyzer", "main"]