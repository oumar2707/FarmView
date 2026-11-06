import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("streamlit") is None,
    reason="Streamlit n'est pas installé dans l'environnement de test.",
)


def test_diagnostic_page_renders_without_app_exception():
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run()
    app.session_state["page"] = "Diagnostic"
    app.run()

    assert not app.exception
