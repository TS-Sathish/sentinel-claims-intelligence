from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def test_all_dashboard_navigation_pages_render():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    navigation = app.radio(key="workspace_navigation")
    pages = navigation.options
    for page in pages:
        app.radio(key="workspace_navigation").set_value(page).run()
        assert not app.exception, f"Dashboard page failed to render: {page}"
