"""Tests for AES-256-GCM + RSA-2048-OAEP encryption."""
from app.core.security import (
    generate_rsa_keypair, export_public_key_pem, load_public_key_pem,
    rsa_encrypt, rsa_decrypt,
    generate_aes_key, aes_encrypt, aes_decrypt,
    hybrid_encrypt, hybrid_decrypt,
    get_server_public_key_pem,
)


def test_aes_roundtrip():
    key = generate_aes_key()
    assert len(key) == 32  # 256-bit
    plaintext = b"gradient payload: [0.1, -0.2, 0.3]"
    enc = aes_encrypt(key, plaintext)
    assert "nonce" in enc and "ciphertext" in enc
    dec = aes_decrypt(key, enc["nonce"], enc["ciphertext"])
    assert dec == plaintext


def test_aes_different_nonce():
    key = generate_aes_key()
    enc1 = aes_encrypt(key, b"same plaintext")
    enc2 = aes_encrypt(key, b"same plaintext")
    assert enc1["nonce"] != enc2["nonce"]  # nonces must be unique


def test_rsa_roundtrip():
    priv, pub = generate_rsa_keypair()
    message = b"aes-session-key-32-bytes-exactly!"[:32]
    ct = rsa_encrypt(pub, message)
    pt = rsa_decrypt(priv, ct)
    assert pt == message


def test_rsa_pem_export_import():
    priv, pub = generate_rsa_keypair()
    pem = export_public_key_pem(pub)
    assert "BEGIN PUBLIC KEY" in pem
    loaded = load_public_key_pem(pem)
    assert loaded is not None


def test_hybrid_encrypt_decrypt():
    priv, pub = generate_rsa_keypair()
    pem = export_public_key_pem(pub)
    payload = b"FL gradient update payload"
    enc = hybrid_encrypt(pem, payload)
    assert "encrypted_key" in enc
    assert "nonce" in enc
    assert "ciphertext" in enc
    dec = hybrid_decrypt(priv, enc["encrypted_key"], enc["nonce"], enc["ciphertext"])
    assert dec == payload


def test_server_keypair_stable():
    """Server keypair is module-level singleton."""
    pem1 = get_server_public_key_pem()
    pem2 = get_server_public_key_pem()
    assert pem1 == pem2
