from src.utils.admin_auth import authenticate_admin


def test_admin_authentication_accepts_configured_credentials():
    assert authenticate_admin("admin@example.test", "secret", "admin@example.test", "secret")


def test_admin_authentication_rejects_wrong_credentials():
    assert not authenticate_admin("admin@example.test", "wrong", "admin@example.test", "secret")


def test_admin_authentication_rejects_missing_configuration():
    assert not authenticate_admin("admin@example.test", "secret", "admin@example.test", None)
