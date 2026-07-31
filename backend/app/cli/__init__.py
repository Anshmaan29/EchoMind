# CLI package initialization
from app.cli.embed import scan_input_path
from app.cli.search import main_async as search_async

__all__ = ["scan_input_path", "search_async"]
