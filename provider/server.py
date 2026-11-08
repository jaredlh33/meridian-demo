"""Meridian provider node.

Runs a REAL local model on this machine's CPU (llama.cpp via
llama-cpp-python) and serves it over an OpenAI-compatible HTTP API.

What it does NOT do (this is the whole point):
  * no external inference API — no third-party gateway, no API keys, no network
    egress to any inference provider. The only network traffic is the
    HTTP server itself answering your requests.
  * no reselling: the model file lives on this disk, its SHA-256 is
    published in the manifest, and anyone can reproduce the output
    from those exact weights with greedy decoding.

Endpoints
  GET  /health                     liveness
  GET  /v1/models                  OpenAI model list (one entry)
  GET  /v1/manifest                model id, weights SHA-256, prices,
                                   and the fingerprint expectation
  POST /v1/fingerprint             run the fixed canary prompt greedily
                                   and return the completion + its hash
  POST /v1/chat/completions        OpenAI-compatible chat completions
                                   (streaming and non-streaming)

Usage:
  python server.py --model ../models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
      --model-id qwen2.5-0.5b --port 8101 \
      --price-in 200 --price-out 500
Prices are micro-USD per million tokens (200 = $0.0002 / 1M input tok).
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


# The one and only inference dependency. It bundles llama.cpp; the model
# file is loaded from local disk and never leaves the machine.
#
# Imported lazily inside Provider.__init__, not at module top:
# importing llama_cpp opens (and never closes) a /dev/null handle,
# which shows up as a ResourceWarning in any process that merely
# imports this module (e.g. the test suite, which stubs the backend).
# main() constructs a Provider at startup, so a missing llama.cpp
# still fails fast there.
def _load_llama():
    from llama_cpp import Llama
    return Llama

# Fixed canary prompt used for the fingerprint check. Deterministic:
# greedy decoding (temperature 0) from fixed weights must reproduce the
# same completion on any machine running the same file.
CANARY_PROMPT = (
    "The capital of France is Paris. The capital of Japan is Tokyo. "
    "The capital of Australia is"
)
CANARY_MAX_TOKENS = 8


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Provider:
    def __init__(self, model_path, model_id, price_in, price_out,
                 n_ctx=2048, n_threads=2):
        self.model_id = model_id
        self.model_path = os.path.abspath(model_path)
        self.weights_sha256 = sha256_file(self.model_path)
        self.weights_bytes = os.path.getsize(self.model_path)
        self.price_in = price_in      # micro-USD per 1M input tokens
        self.price_out = price_out    # micro-USD per 1M output tokens
        self.n_ctx = n_ctx
        self._lock = threading.Lock()
        print(f"[provider] loading {model_path} ...", flush=True)
        self.llm = _load_llama()(model_path=self.model_path, n_ctx=n_ctx,
                                 n_threads=n_threads, verbose=False)
        print("[provider] model loaded, computing fingerprint ...", flush=True)
        out = self._greedy_complete(CANARY_PROMPT, CANARY_MAX_TOKENS)
        self.fingerprint = {
            "canary_prompt": CANARY_PROMPT,
            "canary_max_tokens": CANARY_MAX_TOKENS,
            "expected_output": out,
            "expected_output_sha256": hashlib.sha256(
                out.encode("utf-8")).hexdigest(),
        }
        print(f"[provider] fingerprint: {out!r} "
              f"sha256={self.fingerprint['expected_output_sha256'][:16]}...",
              flush=True)

    def _greedy_complete(self, prompt: str, max_tokens: int) -> str:
        with self._lock:
            res = self.llm(prompt, max_tokens=max_tokens, temperature=0.0,
                           top_p=1.0, top_k=1, echo=False, stream=False)
        return res["choices"][0]["text"]

    def manifest(self) -> dict:
        return {
            "model_id": self.model_id,
            "weights_file": os.path.basename(self.model_path),
            "weights_sha256": self.weights_sha256,
            "weights_bytes": self.weights_bytes,
            "context_tokens": self.n_ctx,
            "max_output_tokens": 512,
            "prices_micro_usd_per_million": {
                "input": self.price_in,
                "output": self.price_out,
            },
            "fingerprint": self.fingerprint,
            # Proof of locality: the weights are a file on this disk.
            "inference": "local-llama.cpp-cpu",
            "upstream": None,
        }

    def chat(self, messages, max_tokens=256, temperature=0.7,
             stream=False):
        """Run one chat completion. Returns (text, prompt_tokens,
        completion_tokens). When stream=True, yields SSE-ready chunks."""
        with self._lock:
            if stream:
                gen = self.llm.create_chat_completion(
                    messages=messages, max_tokens=max_tokens,
                    temperature=temperature, stream=True)
                return gen  # caller iterates; tokens counted afterwards
            res = self.llm.create_chat_completion(
                messages=messages, max_tokens=max_tokens,
                temperature=temperature, stream=False)
        choice = res["choices"][0]
        usage = res.get("usage", {})
        return (choice["message"]["content"],
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0))

    def count_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8")))


PROVIDER: Provider = None  # set in main()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep stdout readable
        sys.stderr.write(
            f"[provider] {self.address_string()} - {fmt % args}\n")

    def _send_json(self, status, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return None

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(200, {"status": "ok",
                                  "model": PROVIDER.model_id})
        elif path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [{
                "id": PROVIDER.model_id, "object": "model",
                "created": int(time.time()), "owned_by": "meridian-local",
            }]})
        elif path == "/v1/manifest":
            self._send_json(200, PROVIDER.manifest())
        else:
            self._send_json(404, {"error": {"message": "no such route",
                                            "code": "not_found"}})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/v1/fingerprint":
            out = PROVIDER._greedy_complete(CANARY_PROMPT,
                                            CANARY_MAX_TOKENS)
            self._send_json(200, {
                "model": PROVIDER.model_id,
                "weights_sha256": PROVIDER.weights_sha256,
                "output": out,
                "output_sha256": hashlib.sha256(
                    out.encode("utf-8")).hexdigest(),
                "expected_output_sha256":
                    PROVIDER.fingerprint["expected_output_sha256"],
                "match": out == PROVIDER.fingerprint["expected_output"],
            })
            return
        if path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "no such route",
                                            "code": "not_found"}})
            return
        body = self._read_json()
        if body is None or "messages" not in body:
            self._send_json(400, {"error": {"message":
                                            "messages required",
                                            "code": "bad_request"}})
            return
        messages = body["messages"]
        max_tokens = int(body.get("max_tokens", 256))
        temperature = float(body.get("temperature", 0.7))
        stream = bool(body.get("stream", False))
        try:
            if stream:
                self._stream_chat(messages, max_tokens, temperature)
            else:
                text, pt, ct = PROVIDER.chat(messages, max_tokens,
                                             temperature, stream=False)
                self._send_json(200, {
                    "id": f"chatcmpl-meridian-{int(time.time()*1000)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": PROVIDER.model_id,
                    "choices": [{"index": 0,
                                 "message": {"role": "assistant",
                                             "content": text},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": pt,
                              "completion_tokens": ct,
                              "total_tokens": pt + ct},
                })
        except Exception as e:  # noqa: BLE001 - surface as 500
            self._send_json(500, {"error": {"message": str(e),
                                            "code": "upstream_error"}})

    def _stream_chat(self, messages, max_tokens, temperature):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        full = []
        try:
            gen = PROVIDER.chat(messages, max_tokens, temperature,
                                stream=True)
            # Hold the model lock for the whole iteration: llama.cpp
            # state is not safe to share between concurrent requests.
            with PROVIDER._lock:
                for chunk in gen:
                    delta = chunk["choices"][0].get("delta", {})
                    piece = delta.get("content", "")
                    if piece:
                        full.append(piece)
                        payload = json.dumps({
                            "id": "chatcmpl-meridian-stream",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": PROVIDER.model_id,
                            "choices": [{"index": 0,
                                         "delta": {"content": piece},
                                         "finish_reason": None}]})
                        self.wfile.write(f"data: {payload}\n\n".encode())
                        self.wfile.flush()
        finally:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()


def main():
    global PROVIDER
    ap = argparse.ArgumentParser(description="Meridian provider node")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--price-in", type=int, default=120,
                    help="micro-USD per 1M input tokens")
    ap.add_argument("--price-out", type=int, default=300,
                    help="micro-USD per 1M output tokens")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    PROVIDER = Provider(args.model, args.model_id, args.price_in,
                        args.price_out, n_threads=args.threads)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[provider] {args.model_id} listening on "
          f"{args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
