"""Shared source-tree and frozen-bundle locations."""
from pathlib import Path
import sys


def project_root():
    return Path(__file__).resolve().parents[2]


def resource_root():
    return Path(getattr(sys,'_MEIPASS',project_root()))
