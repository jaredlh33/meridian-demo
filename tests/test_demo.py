"""Meridian live demo backend tests.

Run:  python -m unittest discover -s tests -v
No model, no network, no API keys needed — the backend is stubbed,
so these run anywhere. The real model path is covered by a live
curl check of demo/server.py (see demo/README.md).

Covered:
  * health endpoint shape + CORS headers
  * health new fields: uptime_seconds, last_request_at (null until
    the first served chat), full canary hash + prompt per model
  * last_request_at is set only by a successfully served chat,
    never by a rejected request
  * chat happy path (verified answer, cost, latency, gates)
  * prompt validation (empty, whitespace, too long, missing, wrong type)
  * max_tokens clamped to 64
  * unknown model rejected
  * verification failure -> 503 with verified:false and NO answer
  * per-IP rate limit -> 429 after the limit
  * CORS preflight (OPTIONS)
  * unknown route -> 404
"""

import importlib.util
import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

_spec = importlib.util.spec_from_file_location(
    "demo_server",
    os.path.join(os.path.dirname(__file__), "..", "demo", "server.py"))
demo_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo_server)


class FakeBackend:
    """Stub DemoBackend: no model, deterministic answers."""

    def __init__(self, fail_verify=False):
        self.fail_verify = fail_verify
        self.seen_max_tokens = []
        self.gates = {
            "weights_sha256_match": True,
            "oath_ok": True,
            "fingerprint_match": True,
            "verified": True,
        }

    def model_ids(self):
        return ["qwen2.5-0.5b", "smollm2-360m"]

    def has_model(self, model_id):
        return model_id in self.model_ids()

    def info(self):
        return [{"model_id": m, "verified": True,
                 "weights_sha256": "ab" * 8,
                 "canary_expected_output_sha256": "cd" * 32,
                 "canary_prompt": demo_server.CANARY_PROMPT,
                 "price_in": 200, "price_out": 500}
                for m in self.model_ids()]

    def ok(self):
        return not self.fail_verify

    def chat(self, model_id, prompt, max_tokens):
        if model_id not in self.model_ids():
            raise KeyError(model_id)
        if self.fail_verify:
            raise demo_server.VerificationFailed(dict(self.gates,
                                                      verified=False))
        self.seen_max_tokens.append(max_tokens)
        return {
            "answer": "Paris",
            "model": model_id,
            "verified": True,
            "gates": dict(self.gates),
            "fingerprint": "cd" * 8,
            "weights_sha256": "ab" * 8,
            "cost_micro_usd": 1,
            "prompt_tokens": 14,
            "completion_tokens": 1,
            "price_in": 200,
            "price_out": 500,
            "max_tokens": max_tokens,
        }


class TestDemoBackend(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        demo_server.BACKEND = self.backend
        demo_server.RATE_LIMITER = demo_server.RateLimiter(limit=10)
        demo_server.REQUEST_COUNT = 0
        demo_server.LAST_REQUEST_AT = None
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0),
                                       demo_server.Handler)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever,
                         daemon=True).start()
        self.addCleanup(self._stop_server)

    def _stop_server(self):
        # shutdown() stops serve_forever; server_close() frees the
        # listening socket. shutdown() alone leaks it (ResourceWarning).
        self.srv.shutdown()
        self.srv.server_close()

    def _call(self, method, path, body=None):
        url = f"http://127.0.0.1:{self.port}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type":
                                              "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read() or b"{}"), r.headers
        except urllib.error.HTTPError as e:
            try:
                raw = e.read()
                try:
                    return e.code, json.loads(raw or b"{}"), e.headers
                except json.JSONDecodeError:
                    return e.code, {}, e.headers
            finally:
                # The error response owns the client socket; reading
                # without closing leaks it (ResourceWarning).
                e.close()

    def _chat(self, **kw):
        body = {"prompt": "What is the capital of France?",
                "model": "qwen2.5-0.5b"}
        body.update(kw)
        return self._call("POST", "/v1/demo/chat", body)

    def test_health_shape_and_cors(self):
        status, body, headers = self._call("GET", "/v1/demo/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual([m["model_id"] for m in body["models"]],
                         ["qwen2.5-0.5b", "smollm2-360m"])
        self.assertTrue(all(m["verified"] for m in body["models"]))
        self.assertIsInstance(body["requests_served"], int)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "*")

    def test_health_new_fields(self):
        status, body, _ = self._call("GET", "/v1/demo/health")
        self.assertEqual(status, 200)
        # uptime: seconds since process start, non-negative int.
        self.assertIsInstance(body["uptime_seconds"], int)
        self.assertGreaterEqual(body["uptime_seconds"], 0)
        # No chat served yet: timestamp is null.
        self.assertIsNone(body["last_request_at"])
        # Each model carries the full canary fingerprint + prompt;
        # the truncated weights hash is unchanged.
        for m in body["models"]:
            self.assertEqual(len(m["canary_expected_output_sha256"]), 64)
            self.assertRegex(m["canary_expected_output_sha256"],
                             r"^[0-9a-f]{64}$")
            self.assertEqual(m["canary_prompt"],
                             demo_server.CANARY_PROMPT)
            self.assertEqual(len(m["weights_sha256"]), 16)

    def test_last_request_at_set_after_chat_only(self):
        import datetime
        _, before, _ = self._call("GET", "/v1/demo/health")
        self.assertIsNone(before["last_request_at"])
        # A rejected request (empty prompt) must NOT set it.
        status, _, _ = self._chat(prompt="")
        self.assertEqual(status, 400)
        _, still_before, _ = self._call("GET", "/v1/demo/health")
        self.assertIsNone(still_before["last_request_at"])
        # A successfully served chat sets it to a parseable UTC time.
        status, _, _ = self._chat()
        self.assertEqual(status, 200)
        _, after, _ = self._call("GET", "/v1/demo/health")
        self.assertEqual(after["requests_served"], 1)
        self.assertIsInstance(after["last_request_at"], str)
        parsed = datetime.datetime.fromisoformat(
            after["last_request_at"])
        self.assertIsNotNone(parsed.tzinfo)
        self.assertGreaterEqual(after["uptime_seconds"], 0)

    def test_chat_happy_path(self):
        status, body, _ = self._chat()
        self.assertEqual(status, 200)
        self.assertEqual(body["answer"], "Paris")
        self.assertEqual(body["model"], "qwen2.5-0.5b")
        self.assertTrue(body["verified"])
        self.assertTrue(body["gates"]["verified"])
        self.assertTrue(body["gates"]["fingerprint_match"])
        self.assertEqual(body["cost_micro_usd"], 1)
        self.assertIsInstance(body["latency_ms"], int)
        self.assertEqual(body["max_tokens"], 32)  # the default

    def test_prompt_validation(self):
        for bad in ("", "   ", "x" * 501, None, 123, ["hi"]):
            with self.subTest(prompt=repr(bad)[:20]):
                if bad is None:
                    status, _, _ = self._call(
                        "POST", "/v1/demo/chat",
                        {"model": "qwen2.5-0.5b"})
                else:
                    status, _body, _ = self._chat(prompt=bad)
                self.assertEqual(status, 400)

    def test_max_tokens_capped_at_64(self):
        status, body, _ = self._chat(max_tokens=500)
        self.assertEqual(status, 200)
        self.assertEqual(body["max_tokens"], 64)
        self.assertEqual(self.backend.seen_max_tokens, [64])

    def test_unknown_model_rejected(self):
        status, body, _ = self._chat(model="gpt-4")
        self.assertEqual(status, 400)
        self.assertIn("unknown model", body["error"]["message"])

    def test_verification_failure_is_503_without_answer(self):
        self.backend.fail_verify = True
        status, body, _ = self._chat()
        self.assertEqual(status, 503)
        self.assertFalse(body["verified"])
        self.assertNotIn("answer", body)
        self.assertEqual(body["error"]["code"], "unverified")

    def test_rate_limit(self):
        demo_server.RATE_LIMITER = demo_server.RateLimiter(limit=3)
        for _ in range(3):
            status, _, _ = self._chat()
            self.assertEqual(status, 200)
        status, body, _ = self._chat()
        self.assertEqual(status, 429)
        self.assertEqual(body["error"]["code"], "rate_limited")

    def test_cors_preflight(self):
        status, _, headers = self._call("OPTIONS", "/v1/demo/chat")
        self.assertEqual(status, 204)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "*")
        self.assertIn("POST", headers.get("Access-Control-Allow-Methods"))

    def test_unknown_route_404(self):
        status, body, _ = self._call("GET", "/nope")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "not_found")


if __name__ == "__main__":
    unittest.main()
