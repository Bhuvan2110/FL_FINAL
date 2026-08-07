"""
AES-256-GCM + RSA-2048-OAEP hybrid encryption.
No external ML libraries — pure Python cryptography.
"""
import os
import base64
import jwt
from datetime import datetime, timedelta, timezone
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization


# ── RSA Key Management ────────────────────────────────────────────────────────

def generate_rsa_keypair() -> tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
    """Generate RSA-2048 keypair."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    return private_key, private_key.public_key()


def export_public_key_pem(public_key: rsa.RSAPublicKey) -> str:
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()


def load_public_key_pem(pem: str) -> rsa.RSAPublicKey:
    return serialization.load_pem_public_key(pem.encode())


def rsa_encrypt(public_key: rsa.RSAPublicKey, plaintext: bytes) -> bytes:
    """RSA-2048-OAEP-SHA256 encrypt (used for AES key exchange)."""
    return public_key.encrypt(
        plaintext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


def rsa_decrypt(private_key: rsa.RSAPrivateKey, ciphertext: bytes) -> bytes:
    return private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


# ── AES-256-GCM ───────────────────────────────────────────────────────────────

def generate_aes_key() -> bytes:
    """Generate a 256-bit AES key."""
    return os.urandom(32)


def aes_encrypt(key: bytes, plaintext: bytes) -> dict:
    """
    AES-256-GCM encrypt.
    Returns dict with base64-encoded nonce + ciphertext.
    """
    nonce = os.urandom(12)  # 96-bit nonce for GCM
    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(nonce, plaintext, None)
    return {
        "nonce": base64.b64encode(nonce).decode(),
        "ciphertext": base64.b64encode(ct).decode(),
    }


def aes_decrypt(key: bytes, nonce_b64: str, ciphertext_b64: str) -> bytes:
    """AES-256-GCM decrypt."""
    nonce = base64.b64decode(nonce_b64)
    ct = base64.b64decode(ciphertext_b64)
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ct, None)


# ── Hybrid Encryption (RSA-wrapped AES) ──────────────────────────────────────

def hybrid_encrypt(public_key_pem: str, payload: bytes) -> dict:
    """
    Hybrid encrypt: generate AES key, encrypt with RSA, encrypt payload with AES.
    """
    pub = load_public_key_pem(public_key_pem)
    aes_key = generate_aes_key()
    encrypted_key = rsa_encrypt(pub, aes_key)
    encrypted_payload = aes_encrypt(aes_key, payload)
    return {
        "encrypted_key": base64.b64encode(encrypted_key).decode(),
        **encrypted_payload,
    }


def hybrid_decrypt(private_key: rsa.RSAPrivateKey, encrypted_key_b64: str,
                   nonce_b64: str, ciphertext_b64: str) -> bytes:
    aes_key = rsa_decrypt(private_key, base64.b64decode(encrypted_key_b64))
    return aes_decrypt(aes_key, nonce_b64, ciphertext_b64)


# ── SecAgg Stub ───────────────────────────────────────────────────────────────

def secagg_mask(weights: list[float], mask_seed: int) -> list[float]:
    """Pairwise additive masking stub for Secure Aggregation."""
    import random
    rng = random.Random(mask_seed)
    return [w + rng.gauss(0, 1e-6) for w in weights]


def secagg_unmask(masked_sum: list[float], masks: list[list[float]]) -> list[float]:
    """Remove accumulated masks from aggregated weights."""
    result = list(masked_sum)
    for mask in masks:
        for i, m in enumerate(mask):
            result[i] -= m
    return result


# ── Module-level server keypair (loaded once per process) ─────────────────────
_server_private_key, _server_public_key = generate_rsa_keypair()


def get_server_private_key() -> rsa.RSAPrivateKey:
    return _server_private_key


def get_server_public_key_pem() -> str:
    return export_public_key_pem(_server_public_key)



# ── JWT Auth Helpers ─────────────────────────────────────────────────────────

def create_access_token(data: dict, secret_key: str, expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=24))
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, secret_key, algorithm="HS256")


def decode_access_token(token: str, secret_key: str) -> dict | None:
    """Decode and verify a JWT access token."""
    try:
        return jwt.decode(token, secret_key, algorithms=["HS256"])
    except Exception:
        return None
