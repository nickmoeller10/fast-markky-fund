# run_app.py
# ======================================================================
# Launcher script for Fast Markky Fund Streamlit application
# ======================================================================
# This script allows you to run the app by clicking the play button
# in PyCharm or other IDEs
# ======================================================================

import subprocess
import sys
import os

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Run streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user.")
    except Exception as e:
        print(f"\n\nError launching application: {e}")
        sys.exit(1)

