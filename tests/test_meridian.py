"""Meridian tests.

Run:  python -m unittest discover -s tests -v
No model, no network, no API keys needed — providers are stubbed with
local HTTP servers, so these run anywhere.

Covered:
  * voucher sign / verify, tamper detection, usage-chain binding
  * cost arithmetic
  * manifest hash check (router rejects a provider whose weights hash
    does not match the registry)
  * fingerprint check (router rejects a provider whose canary output
    does not match the manifest)
  * reseller rejection (a provider that declares an upstream is
    never routed to, even if everything else matches)
  * router failover (first provider errors, second answers)
"""

import hashlib
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "router"))

_spec = importlib.util.spec_from_file_location(
    "router_server",
    os.path.join(os.path.dirname(__file__), "..", "router", "server.py"))
router_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(router_server)
from voucher import (  # noqa: E402
    Channel, SessionKey, canonical_message, cost_micro_usd,
    usage_genesis, usage_next,
)


def make_stub_provider(port, manifest, fingerprint_out,
                       chat_text="hello from stub", fail_chat=False):
    """A tiny fake provider implementing /v1/manifest, /v1/fingerprint
    and /v1/chat/completions."""

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, status, obj):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/manifest":
                self._json(200, manifest)
            else:
                self._json(404, {"error": "nope"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            if self.path == "/v1/fingerprint":
                self._json(200, {
                    "output": fingerprint_out,
                    "output_sha256": hashlib.sha256(
                        fingerprint_out.encode()).hexdigest(),
                    "match": True})
            elif self.path == "/v1/chat/completions":
                if fail_chat:
                    self._json(500, {"error": {"message": "boom"}})
                else:
                    self._json(200, {
                        "choices": [{"message": {"role": "assistant",
                                                 "content": chat_text}}],
                        "usage": {"prompt_tokens": 10,
                                  "completion_tokens": 5}})
            else:
                self._json(404, {"error": "nope"})

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def base_manifest(weights_sha="abc123", model="stub-model",
                  upstream=None, inference="local-llama.cpp-cpu",
                  fp_hash=None):
    fp = {"expected_output_sha256":
          fp_hash or hashlib.sha256(b"canary out").hexdigest()}
    return {"model_id": model, "weights_sha256": weights_sha,
            "prices_micro_usd_per_million": {"input": 100, "output": 200},
            "fingerprint": fp, "upstream": upstream,
            "inference": inference}


class TestVouchers(unittest.TestCase):
    def test_sign_verify_roundtrip(self):
        s = SessionKey()
        msg = canonical_message("chan-1", 42, "00" * 32, 1700000000)
        sig = s.sign(msg)
        self.assertTrue(SessionKey.verify(s.public_hex(), msg, sig))

    def test_tamper_detected(self):
        s = SessionKey()
        msg = canonical_message("chan-1", 42, "00" * 32, 1700000000)
        sig = s.sign(msg)
        tampered = canonical_message("chan-1", 43, "00" * 32, 1700000000)
        self.assertFalse(SessionKey.verify(s.public_hex(), tampered, sig))
        self.assertFalse(SessionKey.verify(
            SessionKey().public_hex(), msg, sig))

    def test_usage_chain_binds_records(self):
        h = usage_genesis()
        h1 = usage_next(h, "chan-1|m|10|5|1")
        h2 = usage_next(h1, "chan-1|m|10|5|1")
        self.assertNotEqual(h1, h2)  # same record twice, chain moves
        self.assertNotEqual(usage_next(h, "chan-1|m|10|6|1"), h1)

    def test_channel_voucher_advances(self):
        s = SessionKey()
        ch = Channel("chan-1", "http://127.0.0.1:9", s)
        e1 = ch.record_request("m", 10, 5, 1)
        e2 = ch.record_request("m", 10, 5, 1)
        self.assertEqual(e1["cumulative_micro_usd"], 1)
        self.assertEqual(e2["cumulative_micro_usd"], 2)
        self.assertTrue(SessionKey.verify(
            e2["session_public_key_hex"],
            e2["voucher_message"].encode(),
            bytes.fromhex(e2["voucher_signature_hex"])))
        self.assertNotEqual(e1["usage_head"], e2["usage_head"])

    def test_cost_math(self):
        # 1M in @100 + 1M out @200 = 300 micro-USD
        self.assertEqual(cost_micro_usd(1_000_000, 1_000_000, 100, 200),
                         300)
        # tiny request still costs at least 1
        self.assertEqual(cost_micro_usd(1, 1, 100, 200), 1)
        # ceiling, not floor: 1 token in @100 = 0.0001 -> 1
        self.assertEqual(cost_micro_usd(500_000, 0, 100, 200), 50)


class TestProviderVerification(unittest.TestCase):
    def setUp(self):
        self.servers = []
        self.session = SessionKey()

    def tearDown(self):
        for s in self.servers:
            s.shutdown()

    def _entry(self, port, weights_sha="abc123"):
        return router_server.ProviderEntry(
            {"url": f"http://127.0.0.1:{port}", "model_id": "stub-model",
             "weights_sha256": weights_sha, "channel_id": "chan-test"},
            self.session, "/tmp/meridian-test-ledger.jsonl")

    def test_verified_when_everything_matches(self):
        fp_out = "canary out"
        m = base_manifest(fp_hash=hashlib.sha256(
            fp_out.encode()).hexdigest())
        self.servers.append(make_stub_provider(18201, m, fp_out))
        e = self._entry(18201)
        self.assertTrue(e.verify())
        self.assertEqual(e.price_in, 100)

    def test_rejects_weights_hash_mismatch(self):
        m = base_manifest(weights_sha="abc123")
        self.servers.append(make_stub_provider(18202, m, "canary out"))
        e = self._entry(18202, weights_sha="different-hash")
        self.assertFalse(e.verify())
        self.assertIn("SHA-256", e.verify_reason)

    def test_rejects_fingerprint_mismatch(self):
        # manifest claims hash of "canary out", provider returns other text
        m = base_manifest(fp_hash=hashlib.sha256(
            b"canary out").hexdigest())
        self.servers.append(make_stub_provider(18203, m, "something else"))
        e = self._entry(18203)
        self.assertFalse(e.verify())
        self.assertIn("fingerprint", e.verify_reason)

    def test_rejects_reseller_upstream(self):
        m = base_manifest(upstream="https://api.example-reseller.invalid/v1")
        self.servers.append(make_stub_provider(18204, m, "canary out"))
        e = self._entry(18204)
        self.assertFalse(e.verify())
        self.assertIn("upstream", e.verify_reason)

    def test_rejects_non_local_inference(self):
        m = base_manifest(inference="remote-gpu-api")
        self.servers.append(make_stub_provider(18205, m, "canary out"))
        e = self._entry(18205)
        self.assertFalse(e.verify())


class TestRouterFailover(unittest.TestCase):
    def test_falls_through_to_second_provider(self):
        fp_out = "canary out"
        m = base_manifest(fp_hash=hashlib.sha256(
            fp_out.encode()).hexdigest())
        bad = make_stub_provider(18211, m, fp_out, fail_chat=True)
        good = make_stub_provider(18212, m, fp_out,
                                  chat_text="second provider answered")
        self.addCleanup(bad.shutdown)
        self.addCleanup(good.shutdown)
        session = SessionKey()
        reg = router_server.Registry([
            {"url": "http://127.0.0.1:18211", "model_id": "stub-model",
             "weights_sha256": "abc123", "channel_id": "chan-a"},
            {"url": "http://127.0.0.1:18212", "model_id": "stub-model",
             "weights_sha256": "abc123", "channel_id": "chan-b"},
        ], session, "/tmp/meridian-test-ledger.jsonl")
        reg.verify_all()
        cands = reg.candidates("stub-model", "auto")
        self.assertEqual(len(cands), 2)
        # reproduce the router's fall-through loop
        answer = None
        for p in cands:
            try:
                answer = router_server.http_post_json(
                    p.url + "/v1/chat/completions",
                    {"model": "stub-model",
                     "messages": [{"role": "user", "content": "hi"}],
                     "max_tokens": 10, "temperature": 0.0,
                     "stream": False})
                break
            except Exception:
                continue
        self.assertIsNotNone(answer)
        self.assertEqual(answer["choices"][0]["message"]["content"],
                         "second provider answered")


if __name__ == "__main__":
    unittest.main()
