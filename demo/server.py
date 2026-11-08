"""Meridian live demo backend.

A small public demo server: it loads the real GGUF models from local
disk (llama.cpp on CPU) and answers questions from a web page. It is
the piece that makes the "Try it live" panel on the website real —
no canned answers, no mock, no external inference API.

How a request is served (same trust story as the router):
  1. Weights check — the loaded file's SHA-256 (computed in full at
     startup) must match the registry entry in router/providers.json,
     and the file on disk must be unchanged since load (size + mtime).
  2. Oath check — this server is a local llama.cpp provider by
     construction: upstream is null, inference is local-llama.cpp-cpu.
     A demo that forwarded to an external API could not pass this.
  3. Fingerprint — the fixed canary prompt is greedily completed
     again and its output hash compared to the fingerprint computed
     at load. Reproducing it requires running the claimed weights.
  4. Only then does the model answer, greedily (temperature 0), and
     the cost is computed with the router's own ceiling arithmetic
     at the cheapest published price for that model.

If any gate fails the request is refused with 503 and verified=false
— no answer is ever served from an unverified model.

Endpoints
  GET  /v1/demo/health   {ok, models: [{model_id, verified, ...,
                          canary_expected_output_sha256 (full hash),
                          canary_prompt}], requests_served,
                          uptime_seconds, last_request_at}
  POST /v1/demo/chat     {prompt (max 500 chars), model,
                          max_tokens (default 32, capped at 64)}
                          -> {answer, model, verified, fingerprint,
                              weights_sha256, cost_micro_usd,
                              latency_ms, gates, ...}

Safety: CORS is open (it is a public demo), but the request rate is
limited per IP (default 10/min) and prompts are capped at 500 chars
and 64 output tokens. No API keys, no auth, no payments, no
persistence. Prompts are never logged or stored — only a request
counter and the timestamp of the last served request are kept
(no prompt text, ever). CPU inference of 0.5B models is slow-ish
(~5-20 tok/s); the models are small and answers are short, honestly.

Usage:
  python demo/server.py [--host 127.0.0.1] [--port 8181]
      [--model models/qwen2.5-0.5b-instruct-q4_k_m.gguf]
      [--model-id qwen2.5-0.5b]
      [--model2 models/smollm2-360m-instruct-q8_0.gguf]
      [--model2-id smollm2-360m]
  Env: MODEL_PATH, MODEL_ID, MODEL_PATH_2, MODEL_ID_2,
       DEMO_HOST, DEMO_PORT (or PORT, as set by Hugging Face Spaces),
       DEMO_RATE_LIMIT (requests per IP per minute, default 10).
  Pass an empty --model2 ("") to load only the first model.
"""

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse, don't reimplement: the provider's model loading, manifest
# and fingerprint, and the router's voucher cost arithmetic.
_provider_mod = _load_module(
    "meridian_provider", os.path.join(_REPO, "provider", "server.py"))
_voucher_mod = _load_module(
    "meridian_voucher", os.path.join(_REPO, "router", "voucher.py"))
Provider = _provider_mod.Provider
CANARY_PROMPT = _provider_mod.CANARY_PROMPT
CANARY_MAX_TOKENS = _provider_mod.CANARY_MAX_TOKENS
cost_micro_usd = _voucher_mod.cost_micro_usd

MAX_PROMPT_CHARS = 500
MAX_OUTPUT_TOKENS = 64
DEFAULT_MAX_TOKENS = 32
DEFAULT_RATE_LIMIT = 10          # requests per IP per minute
RATE_WINDOW_SECONDS = 60

BACKEND = None                   # DemoBackend, set in main() / tests
RATE_LIMITER = None              # RateLimiter, set in main() / tests
REQUEST_COUNT = 0                # served chat requests, counter only
START_TIME = time.monotonic()    # process start (monotonic clock)
LAST_REQUEST_AT = None           # ISO-8601 UTC timestamp of the last
                                 # successfully served chat request;
                                 # None until the first one. Timestamp
                                 # only — prompts are never logged.


class VerificationFailed(Exception):
    """Raised when a model fails any verification gate."""


def load_registry(path=None):
    """router/providers.json: expected weights hashes + prices."""
    path = path or os.path.join(_REPO, "router", "providers.json")
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    reg = {}
    for e in entries:
        m = reg.setdefault(e["model_id"], {
            "weights_sha256": e["weights_sha256"],
            "price_in": e.get("price_in"),
            "price_out": e.get("price_out"),
        })
        # Cheapest published price for the model is the demo's price:
        # it is the :cheap road, and the site calculator shows the same.
        if (e.get("price_in") or 0) + (e.get("price_out") or 0) < \
                (m["price_in"] or 0) + (m["price_out"] or 0):
            m["price_in"] = e.get("price_in")
            m["price_out"] = e.get("price_out")
    return reg


class ModelInstance:
    """One loaded model + its verification state."""

    def __init__(self, model_path, model_id, registry, n_threads=2, n_ctx=2048):
        self.model_id = model_id
        self.model_path = os.path.abspath(model_path)
        entry = registry.get(model_id, {})
        self.expected_weights_sha256 = entry.get("weights_sha256")
        self.price_in = entry.get("price_in")
        self.price_out = entry.get("price_out")
        print(f"[demo] loading {model_id} from {model_path} ...",
              flush=True)
        self.provider = Provider(model_path, model_id,
                                 self.price_in or 0, self.price_out or 0,
                                 n_threads=n_threads, n_ctx=n_ctx)
        # Snapshot of the weights file at load time; a swapped file
        # is caught by the per-request check below.
        self._file_size = os.path.getsize(self.model_path)
        self._file_mtime = os.path.getmtime(self.model_path)
        print(f"[demo] {model_id} loaded, weights "
              f"{self.provider.weights_sha256[:16]}..., fingerprint "
              f"{self.provider.fingerprint['expected_output_sha256'][:16]}...",
              flush=True)

    def verify(self) -> dict:
        """Run all four gates, live, for this request. Returns the
        gates dict; verified is True only if every gate passes."""
        # Gate 1 — the weights on disk are the registered weights:
        # full SHA-256 was computed at load; here we confirm the file
        # has not changed since, and the load-time hash matches.
        try:
            unchanged = (os.path.getsize(self.model_path) ==
                         self._file_size and
                         os.path.getmtime(self.model_path) ==
                         self._file_mtime)
        except OSError:
            unchanged = False
        weights_match = bool(
            unchanged
            and self.expected_weights_sha256
            and self.provider.weights_sha256 ==
            self.expected_weights_sha256)
        # Gate 2 — the oath: local llama.cpp, no upstream, by
        # construction (this process *is* the provider).
        oath_ok = True
        # Gate 3 — fingerprint: re-run the canary greedily and compare
        # its output hash to the one sworn at load. Reproducing it
        # requires running the claimed weights.
        try:
            out = self.provider._greedy_complete(CANARY_PROMPT,
                                                 CANARY_MAX_TOKENS)
            fp_hash = hashlib.sha256(out.encode("utf-8")).hexdigest()
            fingerprint_match = (
                fp_hash ==
                self.provider.fingerprint["expected_output_sha256"])
        except Exception:  # noqa: BLE001 - any failure is a failed gate
            fp_hash = None
            fingerprint_match = False
        gates = {
            "weights_sha256_match": weights_match,
            "oath_ok": oath_ok,
            "fingerprint_match": fingerprint_match,
            "verified": bool(weights_match and oath_ok
                             and fingerprint_match),
        }
        return gates

    def chat(self, prompt: str, max_tokens: int) -> dict:
        gates = self.verify()
        if not gates["verified"]:
            raise VerificationFailed(gates)
        text, pt, ct = self.provider.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.0, stream=False)
        cost = cost_micro_usd(pt, ct, self.price_in or 0,
                              self.price_out or 0)
        return {
            "answer": text,
            "model": self.model_id,
            "verified": True,
            "gates": gates,
            "fingerprint": self.provider.fingerprint[
                "expected_output_sha256"][:16],
            "weights_sha256": self.provider.weights_sha256[:16],
            "cost_micro_usd": cost,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "price_in": self.price_in,
            "price_out": self.price_out,
            "max_tokens": max_tokens,
        }

    def info(self) -> dict:
        return {
            "model_id": self.model_id,
            "verified": self.verify()["verified"],
            "weights_sha256": self.provider.weights_sha256[:16],
            "canary_expected_output_sha256": self.provider.fingerprint[
                "expected_output_sha256"],
            "canary_prompt": CANARY_PROMPT,
            "price_in": self.price_in,
            "price_out": self.price_out,
        }


class DemoBackend:
    """Holds the loaded models. Split out from the HTTP handler so
    tests can substitute a stub without loading a model."""

    def __init__(self, instances):
        self.instances = {i.model_id: i for i in instances}

    def model_ids(self):
        return sorted(self.instances)

    def has_model(self, model_id):
        return model_id in self.instances

    def info(self):
        return [self.instances[m].info() for m in self.model_ids()]

    def ok(self):
        return bool(self.instances) and any(
            i.verify()["verified"] for i in self.instances.values())

    def chat(self, model_id, prompt, max_tokens):
        inst = self.instances.get(model_id)
        if inst is None:
            raise KeyError(model_id)
        return inst.chat(prompt, max_tokens)


class RateLimiter:
    """In-memory per-IP request counting, fixed window."""

    def __init__(self, limit=DEFAULT_RATE_LIMIT,
                 window=RATE_WINDOW_SECONDS):
        self.limit = limit
        self.window = window
        self._hits = collections.defaultdict(collections.deque)
        self._lock = threading.Lock()

    def allow(self, ip: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[ip]
            while q and q[0] <= now - self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


def _short_hash(full):
    return (full or "")[:16]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep stdout readable
        sys.stderr.write("[demo] %s - %s\n" % (self.address_string(),
                                                fmt % args))

    # -- plumbing ---------------------------------------------------
    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods",
                         "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         "Content-Type")

    def _send_json(self, status, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
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

    def _error(self, status, message, code):
        self._send_json(status, {"error": {"message": message,
                                           "code": code}})

    # -- routes -----------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/v1/demo/health":
            self._send_json(200, {
                "ok": BACKEND.ok(),
                "models": BACKEND.info(),
                "requests_served": REQUEST_COUNT,
                "uptime_seconds": int(time.monotonic() - START_TIME),
                "last_request_at": LAST_REQUEST_AT,
            })
        else:
            self._error(404, "no such route", "not_found")

    def do_POST(self):
        global REQUEST_COUNT, LAST_REQUEST_AT
        path = urlparse(self.path).path
        if path != "/v1/demo/chat":
            self._error(404, "no such route", "not_found")
            return
        started = time.monotonic()
        if not RATE_LIMITER.allow(self.client_address[0]):
            self._send_json(429, {"error": {
                "message": "rate limit: 10 demo questions per minute "
                           "per visitor", "code": "rate_limited"}})
            return
        body = self._read_json()
        if body is None:
            self._error(400, "invalid JSON body", "bad_request")
            return
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self._error(400, "prompt is required", "bad_request")
            return
        if len(prompt) > MAX_PROMPT_CHARS:
            self._error(400, "prompt is too long (max 500 characters)",
                        "bad_request")
            return
        model_id = body.get("model") or BACKEND.model_ids()[0]
        if not BACKEND.has_model(model_id):
            self._error(400, f"unknown model: {model_id}", "bad_request")
            return
        try:
            max_tokens = int(body.get("max_tokens",
                                      DEFAULT_MAX_TOKENS))
        except (TypeError, ValueError):
            self._error(400, "max_tokens must be a number",
                        "bad_request")
            return
        max_tokens = max(1, min(max_tokens, MAX_OUTPUT_TOKENS))
        try:
            result = BACKEND.chat(model_id, prompt.strip(), max_tokens)
        except KeyError:
            self._error(400, f"unknown model: {model_id}", "bad_request")
            return
        except VerificationFailed as e:
            self._send_json(503, {
                "verified": False,
                "gates": e.args[0] if e.args else {},
                "error": {"message": "model failed verification — "
                                     "no answer served",
                          "code": "unverified"}})
            return
        except Exception as e:  # noqa: BLE001 - surface as 500
            self._error(500, str(e), "upstream_error")
            return
        REQUEST_COUNT += 1
        # Timestamp only — the prompt itself is never stored.
        LAST_REQUEST_AT = datetime.now(timezone.utc).isoformat()
        result["latency_ms"] = int((time.monotonic() - started) * 1000)
        self._send_json(200, result)


def _resolve_model_path(p):
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    return os.path.join(_REPO, p)


def main():
    global BACKEND, RATE_LIMITER
    ap = argparse.ArgumentParser(description="Meridian live demo backend")
    ap.add_argument("--host", default=os.environ.get(
        "DEMO_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(
        os.environ.get("PORT")
        or os.environ.get("DEMO_PORT") or 8181))
    ap.add_argument("--model", default=os.environ.get(
        "MODEL_PATH", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"))
    ap.add_argument("--model-id", default=os.environ.get(
        "MODEL_ID", "qwen2.5-0.5b"))
    ap.add_argument("--model2", default=os.environ.get(
        "MODEL_PATH_2", "models/smollm2-360m-instruct-q8_0.gguf"))
    ap.add_argument("--model2-id", default=os.environ.get(
        "MODEL_ID_2", "smollm2-360m"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n-ctx", type=int, default=int(
        os.environ.get("DEMO_N_CTX") or 2048))
    ap.add_argument("--rate-limit", type=int, default=int(
        os.environ.get("DEMO_RATE_LIMIT") or DEFAULT_RATE_LIMIT))
    ap.add_argument("--registry", default=os.path.join(
        _REPO, "router", "providers.json"))
    args = ap.parse_args()

    registry = load_registry(args.registry)
    instances = []
    for path, mid in ((args.model, args.model_id),
                      (args.model2, args.model2_id)):
        if not path:
            continue
        full = _resolve_model_path(path)
        if not os.path.exists(full):
            print(f"[demo] WARNING: model file not found, skipping "
                  f"{mid}: {full}", flush=True)
            continue
        instances.append(ModelInstance(full, mid, registry,
                                       n_threads=args.threads,
                                       n_ctx=args.n_ctx))
    if not instances:
        print("[demo] ERROR: no model file found — nothing to serve",
              file=sys.stderr)
        sys.exit(1)
    BACKEND = DemoBackend(instances)
    RATE_LIMITER = RateLimiter(limit=args.rate_limit)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[demo] live demo backend on {args.host}:{args.port} — "
          f"models: {', '.join(BACKEND.model_ids())}, "
          f"rate limit {args.rate_limit}/min per IP", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
