# Contributing to Meridian

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./start.sh   # providers on :8101/:8102/:8103, router on :7171
./stop.sh
```

Models are downloaded on first start into `models/` (git-ignored). No API
keys are needed anywhere — providers run GGUF weights on local CPU.

## Running the demo server locally

```bash
source .venv/bin/activate
python demo/server.py            # defaults to 127.0.0.1:8181
python demo/server.py --port 9000 --model models/smollm2-360m-instruct-q8_0.gguf --model-id smollm2-360m
```

Check it with `curl http://127.0.0.1:8181/v1/demo/health` — every model
listed must show `"verified": true` before you trust the answers.
The same server is what runs on Render behind the website's
"Try it live" panel; see `demo/README.md` for endpoints and limits
(prompt max 500 chars, output max 64 tokens, 10 req/min per IP).

## Tests

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

22 tests (11 in `tests/test_meridian.py`, 11 in `tests/test_demo.py`),
covering vouchers, provider verification, router failover, and the demo
backend (validation, rate limiting, CORS, verification-failure 503).
Tests must pass before a pull request, and major new functionality must
come with tests for it.

## The fingerprint canary is build-sensitive

The fingerprint check greedy-decodes a fixed canary prompt and compares
the output hash to the manifest. **The same weights and the same code
can give different greedy completions on different builds/CPUs** —
the canary fingerprint was observed to differ between a local build and
the Render build. So:

- Any change to model loading, prompting, decoding, or build flags
  (llama.cpp version, compiler flags, thread count, context size)
  **must re-verify the canary** against the live backend's published
  hash (`GET /v1/demo/health` → each model's `fingerprint`) before
  the change ships.
- If the fingerprint changes, update the registry
  (`router/providers.json`) and the website's Registry section to the
  new live value — never hard-code a hash from a different build.
- A provider whose fingerprint does not match its manifest is
  rejected by the router; that is the system working, not a bug.

## Pull requests

- Keep the router OpenAI-compatible (`/v1/chat/completions`, `/v1/models`).
- Keep the voucher format backwards compatible, or bump its version string.
- No external inference APIs, no tokens, no admin keys. If a change needs
  trust, it needs a verification mechanism, not a promise.
- Update the README's comparison table when a documented limitation changes.
