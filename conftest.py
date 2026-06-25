"""Shared pytest configuration."""
import sys
import os

# Ensure the project root is on PYTHONPATH when running pytest from any directory
sys.path.insert(0, os.path.dirname(__file__))
