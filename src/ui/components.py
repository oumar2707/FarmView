"""Reusable Streamlit presentation helpers."""

import html
import math
from pathlib import Path

import streamlit as st


THEME_PATH = Path(__file__).with_name("theme.css")


def inject_theme():
    """Load the shared stylesheet and fail clearly if it is missing."""
    if not THEME_PATH.is_file():
        raise FileNotFoundError(f"Feuille de style FarmView absente : {THEME_PATH}")
    st.markdown(f"<style>{THEME_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_status_badge(status, label):
    """Render a labeled status badge using only known status classes."""
    allowed_statuses = {"healthy", "disease_detected", "uncertain"}
    safe_status = status if status in allowed_statuses else "uncertain"
    st.markdown(
        f'<span class="fv-badge fv-status-{safe_status}">{html.escape(label)}</span>',
        unsafe_allow_html=True,
    )


def load_test_metrics(metrics_path):
    """Load held-out test metrics if a valid report is available."""
    import json

    path = Path(metrics_path)
    if not path.is_file():
        return None
    metrics = json.loads(path.read_text(encoding="utf-8"))
    test_metrics = metrics.get("test") if isinstance(metrics, dict) else None
    accuracy = test_metrics.get("accuracy") if isinstance(test_metrics, dict) else None
    if (
        not isinstance(accuracy, (int, float))
        or not math.isfinite(accuracy)
        or not 0 <= accuracy <= 1
    ):
        raise ValueError(f"La métrique test/accuracy est invalide dans {path}.")
    return metrics
