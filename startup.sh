#!/bin/bash
if [ -d "venv" ]; then
    source venv/bin/activate
fi
python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0
