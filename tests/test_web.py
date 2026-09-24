"""Local browser assets should not mix versions during development."""

import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.request import urlopen

from mini_llm.web import make_handler


def test_local_page_and_script_are_not_cached() -> None:
    app = SimpleNamespace(status=lambda: {"status": "ok"})
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        for path in ("/", "/app.js", "/api/status"):
            with urlopen(base + path, timeout=5) as response:
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
