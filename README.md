# Meridian Demo — Render deploy

Live demo backend for [Meridian](https://meridian-2qtxzg07u-slm-e395.vercel.app/): real local GGUF inference on CPU with per-request fingerprint verification. No external inference API, no keys, no payments.

This repo is the **Render free-tier variant**: one model only (SmolLM2-360M, Q8_0), 512-token context, one inference thread — sized to fit Render's free 512 MB web service. The full two-model version lives in the Meridian repo (`demo/`).

## Deploy on Render (free)

1. Create a **Web Service** from this repo (Docker runtime, **Free** plan).
2. Or use the Blueprint: `render.yaml` in this repo already describes the service.
3. Render sets `$PORT` automatically; the container listens on `0.0.0.0:$PORT`.
4. Health check: `GET /v1/demo/health` — expect `{"ok": true, "models": [{"model_id": "smollm2-360m", "verified": true, ...}]}`.

## Endpoints

- `GET /v1/demo/health` — models and verification status
- `POST /v1/demo/chat` — `{"prompt", "model": "smollm2-360m", "max_tokens"}` (max 64) → real answer with verification gates, cost and latency

Rate limit: 10 requests/minute per IP. Prompts are never logged.

## Notes

- **Free tier sleeps** after 15 minutes idle: the first request after sleep takes ~30–60s while the model loads. Later requests answer in ~1s.
- Build takes a while the first time (llama-cpp-python compiles from source, ~10 min, plus a 386 MB model download).
- The demo server verifies the model on every request (weights SHA-256 + a fresh greedy canary fingerprint). Any failed gate → HTTP 503, no answer.
