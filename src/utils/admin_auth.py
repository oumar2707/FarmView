"""Authentication helpers for the demonstration admin portal."""

import hmac


def authenticate_admin(email, password, expected_email, expected_password):
    """Compare submitted credentials without accepting unset configuration."""
    if not all(
        isinstance(value, str) and value
        for value in (email, password, expected_email, expected_password)
    ):
        return False
    return hmac.compare_digest(email, expected_email) and hmac.compare_digest(
        password, expected_password
    )
