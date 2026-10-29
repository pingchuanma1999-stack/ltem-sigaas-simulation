"""Global navigation helper so pages can open each other (e.g. "show this solution in the explorer")."""
from __future__ import annotations

_window = None


def set_window(w):
    global _window
    _window = w


def navigate(page_id: str, **kwargs):
    """Switch to a page; keyword arguments are passed to the page's `open(**kwargs)` if it has one."""
    if _window is not None:
        _window.navigate(page_id, **kwargs)


def window():
    return _window
