#!/usr/bin/env python3
"""
Simple runner script for wearable data analysis that handles import issues.

This script ensures proper imports and runs the main analysis pipeline.
"""

import os
import sys

# Ensure we can import from the src directory
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')

# Add src to Python path
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now run the main script
if __name__ == "__main__":
    # Import and run main
    from main import main
    main()