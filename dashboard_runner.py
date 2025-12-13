# dashboard_runner.py
# ======================================================================
# Standalone Streamlit app runner for the dashboard
# Run with: streamlit run dashboard_runner.py
# ======================================================================

import streamlit as st
import pickle
import os
import sys

# Add current directory to path
sys.path.insert(0, '.')

from dashboard import render_dashboard

# Load backtest data
if os.path.exists('dashboard_data.pkl'):
    with open('dashboard_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    render_dashboard(
        data['equity_df'],
        data.get('quarterly_df'),
        data['config']
    )
else:
    st.error("""
    **No backtest data available.**
    
    Please run the backtest first using:
    ```bash
    python main.py
    ```
    
    Then select "View interactive dashboard" when prompted.
    """)

