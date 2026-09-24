"""Local web interface backed by the real JeePeeTee checkpoint."""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from dataclasses import asdict
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.evaluation import load_model_for_evaluation
from mini_llm.generation import GenerationConfig, generate
from mini_llm.interactive import DEFAULT_CHECKPOINT, DEFAULT_TOKENIZER, _device_from_name

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"


class JeePeeTeeWebApp:
    def __init__(self, checkpoint: Path, tokenizer_path: Path, device_name: str) -> None:
        self.device = _device_from_name(device_name)
        self.tokenizer = BPETokenizer.load(tokenizer_path)
        self.model, self.payload = load_model_for_evaluation(
            checkpoint, self.tokenizer, device=self.device
        )
        self.checkpoint = checkpoint.resolve()
        self.lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        return {
            "name": "JeePeeTee",
            "device": str(self.device),
            "checkpoint": self.checkpoint.name,
            "global_step": self.payload.get("global_step"),
            "parameters": self.model.parameter_count(),
            "vocabulary_size": self.tokenizer.vocab_size,
            "context_length": self.model.config.context_length,
            "layers": self.model.config.num_layers,
            "heads": self.model.config.num_heads,
            "embedding_dimension": self.model.config.embedding_dim,
            "feed_forward_dimension": self.model.config.feed_forward_dim,
            "mode": "TinyStories completion",
        }

    def tokenize(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return the actual prompt IDs and learned token-plus-position vectors."""

        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        prompt = prompt.strip()
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
        if len(prompt_ids) > self.model.config.context_length:
            raise ValueError("prompt exceeds the model's context length")
        with self.lock, torch.no_grad():
            was_training = self.model.training
            self.model.eval()
            try:
                prompt_vectors = self.model.embeddings(
                    torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
                )[0].detach().float().cpu()
            finally:
                self.model.train(was_training)
        token_norms = torch.linalg.vector_norm(prompt_vectors, dim=-1).tolist()
        tokens = [
            {
                "id": token_id,
                "piece": self.tokenizer.vocabulary[token_id],
                "text": self.tokenizer.decode([token_id], skip_special_tokens=True),
                "position": position,
                "embedding_preview": prompt_vectors[position, :8].tolist(),
                "embedding_norm": token_norms[position],
            }
            for position, token_id in enumerate(prompt_ids)
        ]
        return {"prompt": prompt, "prompt_tokens": tokens}

    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        tokenized = self.tokenize(request)
        prompt = tokenized["prompt"]
        prompt_ids = [token["id"] for token in tokenized["prompt_tokens"]]
        max_new_tokens = int(request.get("max_new_tokens", 24))
        if not 1 <= max_new_tokens <= 64:
            raise ValueError("max_new_tokens must be between 1 and 64")
        strategy = request.get("strategy", "sample")
        if strategy not in ("greedy", "sample"):
            raise ValueError("strategy must be greedy or sample")
        if len(prompt_ids) + max_new_tokens > self.model.config.context_length:
            raise ValueError("prompt and output exceed the model's context length")
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            strategy=strategy,
            temperature=0.8,
            top_k=40 if strategy == "sample" else None,
            top_p=0.9 if strategy == "sample" else None,
            repetition_penalty=1.15,
            seed=42,
            trace_top_n=5,
        )
        with self.lock:
            result = generate(
                self.model,
                self.tokenizer,
                prompt,
                config,
                device=self.device,
            )
        generated_prefix: list[int] = []
        steps: list[dict[str, Any]] = []
        for step in result.steps:
            generated_prefix.append(step.selected_token_id)
            data = asdict(step)
            data["partial_completion"] = self.tokenizer.decode(
                generated_prefix, skip_special_tokens=True
            ).strip()
            data["combined_text"] = self.tokenizer.decode(
                prompt_ids + generated_prefix, skip_special_tokens=True
            )
            steps.append(data)
        return {
            "prompt": prompt,
            "prompt_tokens": tokenized["prompt_tokens"],
            "completion": self.tokenizer.decode(
                result.generated_token_ids, skip_special_tokens=True
            ).strip(),
            "finish_reason": result.finish_reason,
            "generated_token_count": len(result.generated_token_ids),
            "settings": asdict(config),
            "steps": steps,
        }


def make_handler(app: JeePeeTeeWebApp) -> type[SimpleHTTPRequestHandler]:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

        def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def end_headers(self) -> None:
            # HTML and JS must stay in sync while the local UI is being updated.
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if urlparse(self.path).path == "/api/status":
                self._json(app.status())
                return
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            if route not in ("/api/generate", "/api/tokenize"):
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 32_768:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request must be a JSON object")
                result = (
                    app.tokenize(payload)
                    if route == "/api/tokenize"
                    else app.complete(payload)
                )
                self._json(result)
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:  # pragma: no cover - defensive HTTP boundary
                self.log_error("generation failed: %s", error)
                self._json({"error": "generation failed"}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[web] {self.address_string()} - {format % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local JeePeeTee web visualizer")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if not WEB_ROOT.is_dir():
        raise FileNotFoundError(f"web assets not found: {WEB_ROOT}")
    app = JeePeeTeeWebApp(args.checkpoint, args.tokenizer, args.device)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    url = f"http://{args.host}:{args.port}"
    print(f"JeePeeTee Lab ready at {url}")
    print("Press Ctrl+C to stop.")
    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping JeePeeTee Lab.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
