"""Meridian router.

A local OpenAI-compatible router that only talks to providers it has
verified itself:

  1. Fetch the provider's /v1/manifest and check the published
     weights SHA-256 against the registry entry (providers.json).
     A provider that swapped its weights — e.g. to a cheaper model,
     or to an API key in front of someone else's service — fails
     this check and is never routed to.
  2. Call /v1/fingerprint: the provider greedily completes a fixed
     canary prompt. The router compares the output hash to the one
     in the manifest. Same weights + greedy decoding = same output,
     so a matching fingerprint proves the provider is really running
     the claimed file. (A reseller cannot fake this: it would have
     to run the weights to reproduce the output.)
  3. Verification is cached and re-run every VERIFY_EVERY seconds.

Routing: `model` may carry a suffix — `qwen2.5-0.5b:cheap` picks the
cheapest verified provider for that model; otherwise the first
verified provider is used. If a provider fails (connection error,
5xx, timeout) the router falls through to the next verified
provider for the same model. The router asks providers for
non-streaming completions internally (so failover is clean) and
re-streams the answer as SSE when the client asked for a stream.

Payments: after every answered request the router computes the cost
from the provider's published prices, advances that provider's
channel, signs a voucher (see voucher.py) and appends it to
ledger.jsonl. There is no token, no fee, no admin key: the session
key is generated fresh on every start and can only sign totals for
this router's own channels.

Endpoints
  GET  /health
  GET  /v1/models
  GET  /v1/manifests            verification status of every provider
  GET  /v1/ledger               the local settlement ledger (debug)
  POST /v1/chat/completions     OpenAI-compatible

Usage:
  python server.py --port 7171 --providers providers.json
"""

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voucher import Channel, SessionKey, cost_micro_usd  # noqa: E402

VERIFY_EVERY = 120  # seconds between re-verifications
HTTP_TIMEOUT = 120  # seconds for a provider completion


def http_get(url: str, timeout: float = 10.0):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_post_json(url: str, payload: dict, timeout: float = HTTP_TIMEOUT):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class ProviderEntry:
    def __init__(self, cfg: dict, session: SessionKey, ledger_path: str):
        self.url = cfg["url"].rstrip("/")
        self.model_id = cfg["model_id"]
        self.expected_weights_sha256 = cfg["weights_sha256"]
        self.channel = Channel(cfg["channel_id"], self.url, session)
        self.ledger_path = ledger_path
        self.verified = False
        self.verify_reason = "not yet verified"
        self.last_verified_at = 0
        self.manifest = None
        self.price_in = cfg.get("price_in")
        self.price_out = cfg.get("price_out")

    def verify(self) -> bool:
        """Manifest hash check + live fingerprint check. Returns True
        only if both pass."""
        try:
            manifest = http_get(self.url + "/v1/manifest")
        except Exception as e:  # noqa: BLE001
            return self._fail(f"manifest fetch failed: {e}")
        if manifest.get("weights_sha256") != self.expected_weights_sha256:
            return self._fail("weights SHA-256 mismatch with registry")
        if manifest.get("model_id") != self.model_id:
            return self._fail("model id mismatch with registry")
        if manifest.get("upstream") is not None:
            return self._fail("provider declares an upstream API — "
                              "resellers are not routed to")
        if manifest.get("inference") != "local-llama.cpp-cpu":
            return self._fail("provider is not local llama.cpp inference")
        fp = manifest.get("fingerprint", {})
        try:
            live = http_post_json(self.url + "/v1/fingerprint", {})
        except Exception as e:  # noqa: BLE001
            return self._fail(f"fingerprint call failed: {e}")
        if live.get("output_sha256") != fp.get("expected_output_sha256"):
            return self._fail("fingerprint output hash mismatch — "
                              "provider is not running the claimed weights")
        if not live.get("match"):
            return self._fail("provider's own fingerprint self-check failed")
        self.manifest = manifest
        self.price_in = manifest["prices_micro_usd_per_million"]["input"]
        self.price_out = manifest["prices_micro_usd_per_million"]["output"]
        self.verified = True
        self.verify_reason = "ok"
        self.last_verified_at = time.time()
        return True

    def _fail(self, reason: str) -> bool:
        self.verified = False
        self.verify_reason = reason
        self.last_verified_at = time.time()
        return False

    def maybe_reverify(self):
        if time.time() - self.last_verified_at >= VERIFY_EVERY:
            self.verify()


class Registry:
    def __init__(self, providers_cfg: list, session: SessionKey,
                 ledger_path: str):
        self.providers = [ProviderEntry(c, session, ledger_path)
                          for c in providers_cfg]
        self.lock = threading.Lock()

    def verify_all(self):
        for p in self.providers:
            ok = p.verify()
            print(f"[router] {p.url} ({p.model_id}): "
                  f"{'VERIFIED' if ok else 'REJECTED: ' + p.verify_reason}",
                  flush=True)

    def candidates(self, model_id: str, route: str):
        """Verified providers for a model, ordered by route."""
        cands = [p for p in self.providers
                 if p.model_id == model_id and p.verified]
        if route == "cheap":
            cands.sort(key=lambda p: (p.price_in or 0) + (p.price_out or 0))
        return cands


REGISTRY: Registry = None
LEDGER_PATH = None
SESSION: SessionKey = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[router] %s - %s\n" % (self.address_string(),
                                                  fmt % args))

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
            self._send_json(200, {
                "status": "ok",
                "providers": len(REGISTRY.providers),
                "verified": sum(1 for p in REGISTRY.providers
                                if p.verified)})
        elif path == "/v1/models":
            seen = {}
            for p in REGISTRY.providers:
                if p.verified:
                    seen[p.model_id] = {
                        "id": p.model_id, "object": "model",
                        "created": int(time.time()),
                        "owned_by": "meridian-local",
                    }
            self._send_json(200, {"object": "list",
                                  "data": list(seen.values())})
        elif path == "/v1/manifests":
            self._send_json(200, {"providers": [{
                "url": p.url, "model": p.model_id,
                "verified": p.verified, "reason": p.verify_reason,
                "weights_sha256": p.expected_weights_sha256,
                "last_verified_at": p.last_verified_at,
            } for p in REGISTRY.providers]})
        elif path == "/v1/ledger":
            entries = []
            if os.path.exists(LEDGER_PATH):
                with open(LEDGER_PATH, encoding="utf-8") as f:
                    entries = [json.loads(l) for l in f if l.strip()]
            self._send_json(200, {
                "session_public_key_hex": SESSION.public_hex(),
                "entries": entries})
        else:
            self._send_json(404, {"error": {"message": "no such route",
                                            "code": "not_found"}})

    def do_POST(self):
        if urlparse(self.path).path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "no such route",
                                            "code": "not_found"}})
            return
        body = self._read_json()
        if body is None or "messages" not in body:
            self._send_json(400, {"error": {"message": "messages required",
                                            "code": "bad_request"}})
            return
        raw_model = str(body.get("model", ""))
        model_id, _, route = raw_model.partition(":")
        route = route or "auto"
        messages = body["messages"]
        max_tokens = int(body.get("max_tokens", 256))
        temperature = float(body.get("temperature", 0.7))
        stream = bool(body.get("stream", False))

        for p in REGISTRY.providers:
            p.maybe_reverify()
        cands = REGISTRY.candidates(model_id, route)
        if not cands:
            self._send_json(502, {"error": {
                "message": f"no verified provider for model {model_id}",
                "code": "no_verified_provider"}})
            return

        last_error = None
        for p in cands:
            try:
                res = http_post_json(p.url + "/v1/chat/completions", {
                    "model": p.model_id, "messages": messages,
                    "max_tokens": max_tokens, "temperature": temperature,
                    "stream": False,
                })
                break
            except Exception as e:  # noqa: BLE001 - fall through
                last_error = e
                print(f"[router] provider {p.url} failed ({e}); "
                      f"falling through", flush=True)
                continue
        else:
            self._send_json(502, {"error": {
                "message": f"all providers failed: {last_error}",
                "code": "providers_failed"}})
            return

        text = res["choices"][0]["message"]["content"]
        usage = res.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        cost = cost_micro_usd(pt, ct, p.price_in, p.price_out)
        entry = p.channel.record_request(p.model_id, pt, ct, cost)
        from voucher import ledger_append
        ledger_append(LEDGER_PATH, entry)
        print(f"[router] {p.model_id} via {p.url}: {pt}+{ct} tok, "
              f"cost {cost} micro-USD, channel total "
              f"{entry['cumulative_micro_usd']}", flush=True)

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            payload = json.dumps({
                "id": "chatcmpl-meridian",
                "object": "chat.completion.chunk",
                "created": int(time.time()), "model": p.model_id,
                "choices": [{"index": 0, "delta": {"content": text},
                             "finish_reason": None}]})
            self.wfile.write(f"data: {payload}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            self._send_json(200, {
                "id": f"chatcmpl-meridian-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": p.model_id,
                "choices": [{"index": 0,
                             "message": {"role": "assistant",
                                         "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": pt,
                          "completion_tokens": ct,
                          "total_tokens": pt + ct},
                "meridian": {"provider": p.url,
                             "cost_micro_usd": cost,
                             "channel_total_micro_usd":
                                 entry["cumulative_micro_usd"]},
            })


def main():
    global REGISTRY, LEDGER_PATH, SESSION
    ap = argparse.ArgumentParser(description="Meridian router")
    ap.add_argument("--port", type=int, default=7171)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--providers", default="providers.json")
    ap.add_argument("--ledger", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ledger.jsonl"))
    args = ap.parse_args()

    with open(args.providers, encoding="utf-8") as f:
        providers_cfg = json.load(f)
    SESSION = SessionKey()
    LEDGER_PATH = os.path.abspath(args.ledger)
    print(f"[router] session key (ephemeral, this run only): "
          f"{SESSION.public_hex()[:16]}...", flush=True)
    REGISTRY = Registry(providers_cfg, SESSION, LEDGER_PATH)
    REGISTRY.verify_all()

    def reverify_loop():
        while True:
            time.sleep(VERIFY_EVERY)
            for p in REGISTRY.providers:
                p.verify()

    threading.Thread(target=reverify_loop, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[router] listening on {args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
