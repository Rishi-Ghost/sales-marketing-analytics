#!/bin/bash
cd "$(dirname "$0")"
streamlit run app/dashboard.py --server.port 8501 --server.headless true
