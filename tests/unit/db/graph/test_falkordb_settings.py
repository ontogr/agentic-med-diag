"""URL and environment validation without contacting a FalkorDB server."""

import pytest
from pydantic import ValidationError

from am_diag.db.graph.settings import FalkorDBSettings


class TestFalkorDBSettings:
    """Cover secret handling, URL precedence and invalid resource limits."""

    def test_decodes_tls_url_and_hides_password(self):
        """Decode URL components without exposing the password in repr."""
        settings = FalkorDBSettings.from_url(
            "rediss://user:p%40ss@host:6380/my%20graph"
        )
        assert (
            settings.host,
            settings.port,
            settings.username,
            settings.graph,
            settings.ssl,
        ) == (
            "host",
            6380,
            "user",
            "my graph",
            True,
        )
        assert settings.password.get_secret_value() == "p@ss"
        assert "p@ss" not in repr(settings)

    def test_explicit_overrides_win_over_url_and_environment(self, monkeypatch):
        """Explicit settings take precedence over both URL and environment."""
        monkeypatch.setenv("FALKORDB_HOST", "environment")
        settings = FalkorDBSettings.from_url(
            "redis://url/g", host="explicit", graph="override"
        )
        assert settings.host == "explicit"
        assert settings.graph == "override"

    def test_loads_secret_and_limits_from_environment(self, monkeypatch):
        """Read secrets and write limits from the FalkorDB environment prefix."""
        monkeypatch.setenv("FALKORDB_PASSWORD", "secret")
        monkeypatch.setenv("FALKORDB_BATCH_SIZE", "25")
        settings = FalkorDBSettings()
        assert settings.password.get_secret_value() == "secret"
        assert settings.batch_size == 25

    @pytest.mark.parametrize(
        "url",
        [
            "http://host",
            "redis://",
            "redis://h:0",
            "redis://h:65536",
            "redis://h?db=1",
            "redis://h#x",
        ],
    )
    def test_rejects_invalid_urls(self, url):
        """Reject unsupported schemes, absent hosts and invalid ports."""
        with pytest.raises(ValueError):
            FalkorDBSettings.from_url(url)

    @pytest.mark.parametrize(
        "values",
        [
            {"port": 0},
            {"port": 65536},
            {"query_timeout_ms": 0},
            {"max_connections": 0},
            {"batch_size": 0},
            {"graph": ""},
            {"host": ""},
        ],
    )
    def test_rejects_invalid_limits(self, values):
        """Reject invalid connection limits before constructing a driver."""
        with pytest.raises(ValidationError):
            FalkorDBSettings(**values)
