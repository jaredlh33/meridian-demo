"""Meridian vouchers: signed running totals for micropayments.

Design (original to Meridian — deliberately NOT Orrery's layout):
  * Money is counted in micro-USD (integer), per payment channel.
  * A voucher is a canonical UTF-8 string, signed with Ed25519:

        meridian-voucher/1
        channel=<channel id>
        total_micro_usd=<cumulative integer>
        usage=<sha256 hex of the usage chain head>
        issued=<unix seconds>

    Lines are joined with "\\n" and the signature covers the whole
    string. No binary packing, no length prefixes — anyone can verify
    a voucher with a text editor and a public key.

  * The usage chain binds every voucher to the work it pays for:
        h0 = sha256("meridian-usage/1")
        hn = sha256(h(n-1) || record_n)
    where record_n is "<channel>|<model>|<in tok>|<out tok>|<cost>"
    in plain text. A provider that redeems a voucher implicitly
    agrees with the usage history it commits to.

  * There is no token, no protocol fee, no admin key. The router's
    session key is generated fresh at startup; it can only ever sign
    totals for its own channels. How this maps to on-chain settlement
    later: replace the local ledger with a Solana program that holds
    channel deposits and accepts exactly this voucher format — the
    message string and the usage chain are already chain-ready, and
    the only on-chain check needed is `ed25519_verify(session_key,
    message, signature)`.

Ledger: JSON lines, one record per signed voucher, at
router/ledger.jsonl. It is the settlement record of this prototype.
"""

import hashlib
import json
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding, PublicFormat,
)

DOMAIN = "meridian-voucher/1"
USAGE_DOMAIN = b"meridian-usage/1"


def canonical_message(channel: str, total_micro_usd: int,
                      usage_hex: str, issued: int) -> bytes:
    text = (
        f"{DOMAIN}\n"
        f"channel={channel}\n"
        f"total_micro_usd={total_micro_usd}\n"
        f"usage={usage_hex}\n"
        f"issued={issued}"
    )
    return text.encode("utf-8")


def usage_genesis() -> bytes:
    return hashlib.sha256(USAGE_DOMAIN).digest()


def usage_next(prev: bytes, record: str) -> bytes:
    return hashlib.sha256(prev + record.encode("utf-8")).digest()


class SessionKey:
    """One ephemeral Ed25519 keypair per router run."""

    def __init__(self, private: Ed25519PrivateKey | None = None):
        self.private = private or Ed25519PrivateKey.generate()
        self.public = self.private.public_key()

    def public_hex(self) -> str:
        return self.public.public_bytes(
            Encoding.Raw, PublicFormat.Raw).hex()

    def sign(self, message: bytes) -> bytes:
        return self.private.sign(message)

    @staticmethod
    def verify(public_hex: str, message: bytes, signature: bytes) -> bool:
        try:
            pub = Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(public_hex))
            pub.verify(signature, message)
            return True
        except Exception:  # noqa: BLE001 - any failure is "invalid"
            return False


class Channel:
    """One payment channel to one provider: a running total, a usage
    chain, and a signed voucher after every request."""

    def __init__(self, channel_id: str, provider_url: str,
                 session: SessionKey):
        self.channel_id = channel_id
        self.provider_url = provider_url
        self.session = session
        self.total_micro_usd = 0
        self.usage_head = usage_genesis()
        self.vouchers = 0

    def record_request(self, model: str, prompt_tokens: int,
                       completion_tokens: int, cost_micro_usd: int) -> dict:
        record = (f"{self.channel_id}|{model}|{prompt_tokens}|"
                  f"{completion_tokens}|{cost_micro_usd}")
        self.usage_head = usage_next(self.usage_head, record)
        self.total_micro_usd += cost_micro_usd
        self.vouchers += 1
        issued = int(time.time())
        msg = canonical_message(self.channel_id, self.total_micro_usd,
                                self.usage_head.hex(), issued)
        sig = self.session.sign(msg)
        return {
            "channel": self.channel_id,
            "provider": self.provider_url,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_micro_usd": cost_micro_usd,
            "cumulative_micro_usd": self.total_micro_usd,
            "usage_head": self.usage_head.hex(),
            "voucher_message": msg.decode("utf-8"),
            "voucher_signature_hex": sig.hex(),
            "session_public_key_hex": self.session.public_hex(),
            "issued": issued,
        }


def cost_micro_usd(prompt_tokens: int, completion_tokens: int,
                   price_in: int, price_out: int) -> int:
    """Ceiling of (in*pin + out*pout) / 1e6, prices in micro-USD per
    million tokens. At least 1 micro-USD per answered request so a
    voucher always moves forward."""
    raw = prompt_tokens * price_in + completion_tokens * price_out
    cost = -(-raw // 1_000_000)
    return max(cost, 1)


def ledger_append(path: str, entry: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
