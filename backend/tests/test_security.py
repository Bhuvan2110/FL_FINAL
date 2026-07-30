"""Tests for security utilities."""
from app.core.security import get_server_public_key_pem


def test_get_server_public_key_pem():
    """Test that server public key is retrievable."""
    public_key_pem = get_server_public_key_pem()
    
    assert isinstance(public_key_pem, str)
    assert len(public_key_pem) > 0
    assert "BEGIN" in public_key_pem  # PEM format has BEGIN


def test_public_key_format():
    """Test that public key is valid PEM format."""
    public_key_pem = get_server_public_key_pem()
    
    # PEM keys should have these markers
    assert "BEGIN PUBLIC KEY" in public_key_pem or "BEGIN RSA PUBLIC KEY" in public_key_pem
    assert "END PUBLIC KEY" in public_key_pem or "END RSA PUBLIC KEY" in public_key_pem


def test_public_key_consistency():
    """Test that public key is consistent across calls."""
    key1 = get_server_public_key_pem()
    key2 = get_server_public_key_pem()
    
    assert key1 == key2


def test_public_key_not_empty():
    """Test that public key is not empty."""
    public_key_pem = get_server_public_key_pem()
    
    assert len(public_key_pem) > 50  # PEM keys are reasonably long
