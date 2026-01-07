# utils.py
import streamlit as st
import numpy as np

def safe_rerun():
    """Force le rechargement de la page de manière compatible."""
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

def apply_window_level(image, window, level):
    """Applique le fenêtrage (Window/Level) sur une image."""
    win = max(0.001, window)
    vmin, vmax = level - win/2, level + win/2
    return np.clip((image - vmin)/(vmax - vmin), 0, 1)

def inject_css():
    """Injecte le style CSS de l'application."""
    st.markdown("""
        <style>
            div[data-baseweb="tab-list"] {
                display: flex !important; flex-wrap: nowrap !important; overflow-x: auto !important;
                white-space: nowrap !important; gap: 8px; padding-bottom: 8px; width: 100%;
                scrollbar-width: thin; scrollbar-color: #4f46e5 #e0e7ff;
            }
            div[data-baseweb="tab"] { flex: 0 0 auto !important; min-width: fit-content !important; }
            div[data-baseweb="tab-list"]::-webkit-scrollbar { height: 10px !important; display: block !important; }
            div[data-baseweb="tab-list"]::-webkit-scrollbar-track { background: #e0e7ff !important; border-radius: 4px; }
            div[data-baseweb="tab-list"]::-webkit-scrollbar-thumb { background-color: #4f46e5 !important; border-radius: 10px; border: 2px solid #e0e7ff; }
            
            .tr-alert-box {
                background-color: #fee2e2; border-left: 5px solid #ef4444; padding: 15px;
                border-radius: 5px; color: #7f1d1d; font-weight: bold; margin-bottom: 15px; margin-top: 5px;
            }
            .opt-box {
                background-color: #eff6ff; border: 1px solid #bfdbfe; border-radius: 5px;
                padding: 10px; text-align: center; margin-bottom: 5px; color: #1e3a8a;
            }
        </style>
    """, unsafe_allow_html=True)