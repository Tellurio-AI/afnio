import builtins
import json
import tempfile

import pytest

from afnio.tellurio.consent import check_consent
from afnio.tellurio.utils import get_config_path


class TestCheckConsent:
    """Test suite for check_consent function."""

    def test_env_true(self, monkeypatch):
        """
        Test that check_consent passes when ALLOW_API_KEY_SHARING is 'true' in env.
        """
        monkeypatch.setenv("ALLOW_API_KEY_SHARING", "true")
        # Should not raise
        check_consent()

    def test_env_invalid(self, monkeypatch):
        """
        Test that check_consent raises ValueError for invalid env value.
        """
        monkeypatch.setenv("ALLOW_API_KEY_SHARING", "maybe")
        with pytest.raises(ValueError):
            check_consent()

    def test_config_true(self, monkeypatch):
        """
        Test that check_consent passes when config file allows sharing.
        """
        config_path = get_config_path()
        with open(config_path, "w") as f:
            json.dump({"allow_api_key_sharing": True}, f)
        monkeypatch.delenv("ALLOW_API_KEY_SHARING", raising=False)
        # Should not raise
        check_consent()

    def test_prompt_yes(self, monkeypatch):
        """
        Test that check_consent passes when user inputs 'yes' interactively.
        """
        config_path = get_config_path()
        with open(config_path, "w") as f:
            json.dump({"allow_api_key_sharing": "not-true"}, f)
        monkeypatch.setattr("afnio.tellurio.consent._is_interactive", lambda: True)
        monkeypatch.delenv("ALLOW_API_KEY_SHARING", raising=False)
        monkeypatch.setattr(builtins, "input", lambda _: "yes")
        # Should not raise
        check_consent()

    def test_prompt_always(self, monkeypatch):
        """
        Test that check_consent passes and updates config when user inputs 'always'.
        """
        config_path = get_config_path()
        with open(config_path, "w") as f:
            json.dump({"allow_api_key_sharing": "not-true"}, f)
        monkeypatch.setattr("afnio.tellurio.consent._is_interactive", lambda: True)
        monkeypatch.delenv("ALLOW_API_KEY_SHARING", raising=False)
        monkeypatch.setattr(builtins, "input", lambda _: "always")
        check_consent()
        with open(config_path) as f:
            config = json.load(f)
        assert config["allow_api_key_sharing"] is True

    def test_prompt_no(self, monkeypatch):
        """
        Test that check_consent raises when user inputs 'no' interactively.
        """
        # Simulate interactive environment
        config_path = get_config_path()
        with open(config_path, "w") as f:
            json.dump({"allow_api_key_sharing": "not-true"}, f)
        monkeypatch.setattr("afnio.tellurio.consent._is_interactive", lambda: True)
        monkeypatch.delenv("ALLOW_API_KEY_SHARING", raising=False)
        monkeypatch.setattr(builtins, "input", lambda _: "no")
        with pytest.raises(
            RuntimeError,
            match="Consent to share your LM API key was not given. Aborting operation.",
        ):
            check_consent()

    def test_noninteractive(self, monkeypatch):
        """
        Test that check_consent raises in non-interactive environment with no consent.
        """
        # Simulate non-interactive environment
        monkeypatch.setattr("afnio.tellurio.consent._is_interactive", lambda: False)
        monkeypatch.delenv("ALLOW_API_KEY_SHARING", raising=False)
        monkeypatch.setattr(
            "afnio.tellurio.utils.get_config_path", lambda: tempfile.mktemp()
        )
        with pytest.raises(
            RuntimeError,
            match=(
                "Consent to share your API key with the server is required, "
                "but no consent was found and the environment is non-interactive. "
                "Set ALLOW_API_KEY_SHARING=True or update ~/.tellurio.config to allow."
            ),
        ):
            check_consent()
