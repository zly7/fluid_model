"""
Simplified Streamlit app for gas pipeline data visualization.
This is the main entry point that calls the multi-variable viewer.
"""

import sys
from pathlib import Path

# Add current directory to path for proper imports
sys.path.append(str(Path(__file__).parent))

from multi_variable_viewer import run_multi_variable_viewer

def run_dashboard():
    """Entry point for running the dashboard."""
    run_multi_variable_viewer()

def main():
    """Main function for compatibility."""
    run_multi_variable_viewer()

if __name__ == "__main__":
    run_multi_variable_viewer()