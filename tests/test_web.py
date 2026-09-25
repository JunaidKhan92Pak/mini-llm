"""Local browser assets should not mix versions during development."""

import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.request import urlopen

import torch

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.config import ModelConfig
from mini_llm.model import JeePeeTee
from mini_llm.web import JeePeeTeeWebApp, make_handler


def test_inside_step_embedding_values_match_model_tables() -> None:
    tokenizer = BPETokenizer.train(
        ["Once upon a time, two friends went home."] * 4,
        vocab_size=300,
        min_frequency=1,
    )
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=64,
        embedding_dim=16,
        num_layers=1,
        num_heads=4,
        feed_forward_dim=32,
        dropout=0.4,
    )
    model = JeePeeTee(config)
    model.train()
    app = JeePeeTeeWebApp.__new__(JeePeeTeeWebApp)
    app.device = torch.device("cpu")
    app.tokenizer = tokenizer
    app.model = model
    app.lock = threading.Lock()

    result = app.tokenize({"prompt": "Once upon a time"})

    assert model.training
    for token in result["prompt_tokens"]:
        assert len(token["embedding_vector"]) == config.embedding_dim
        combined = torch.tensor(token["embedding_vector"])
        token_part = torch.tensor(token["token_embedding_preview"])
        position_part = torch.tensor(token["position_embedding_preview"])
        torch.testing.assert_close(combined[:8], token_part + position_part)
        torch.testing.assert_close(
            torch.linalg.vector_norm(combined),
            torch.tensor(token["embedding_norm"]),
        )

    generated = app.complete(
        {"prompt": "Once upon a time", "strategy": "greedy", "max_new_tokens": 1}
    )
    step = generated["steps"][0]
    parts = step["embedding_components"]
    assert len(parts["embedding_vector"]) == config.embedding_dim
    torch.testing.assert_close(
        torch.tensor(parts["embedding_vector"][:8]),
        torch.tensor(parts["token_embedding_preview"])
        + torch.tensor(parts["position_embedding_preview"]),
    )
    torch.testing.assert_close(
        torch.tensor(parts["embedding_vector"][:8]),
        torch.tensor(step["embedding_preview"]),
    )


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
