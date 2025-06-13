
def get_current_path() -> str:
    """Get the current file path."""
    import os
    return os.path.dirname(os.path.abspath(__file__))