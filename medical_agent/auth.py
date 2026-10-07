"""Signed, expiring local demo identities. Replace demo login with OIDC in a real service."""

import base64
import hashlib
import hmac
import json
import time


def issue_token(user, secret):
    payload = base64.urlsafe_b64encode(
        json.dumps({"sub": user, "exp": int(time.time()) + 3600}).encode()
    ).decode()
    signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return payload + "." + signature


def verify_token(token, secret):
    try:
        payload, signature = token.split(".")
        expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("invalid signature")
        identity = json.loads(base64.urlsafe_b64decode(payload))
        if identity["exp"] <= time.time() or identity["sub"] not in {"alice", "bob"}:
            raise ValueError("invalid identity")
        return identity["sub"]
    except (ValueError, KeyError, TypeError) as exc:
        raise PermissionError("invalid or expired token") from exc
