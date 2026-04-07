"""Pytest configuration — ensures project root is on sys.path."""

import sys
import os

# Add project root to sys.path so imports work without hacks in each test file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
