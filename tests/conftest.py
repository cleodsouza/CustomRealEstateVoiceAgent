"""Provide dummy credentials before config.py loads — it fails fast without them.

conftest runs before any test module imports config, and load_dotenv() never
overrides variables that are already set, so tests are deterministic whether
or not a real .env exists.
"""
import os

_DUMMY_ENV = {
    "SARVAM_API_KEY": "test-sarvam-key",
    "DEEPGRAM_API_KEY": "test-deepgram-key",
    "VOBIZ_AUTH_ID": "test-vobiz-id",
    "VOBIZ_AUTH_TOKEN": "test-vobiz-token",
    "VOBIZ_WHATSAPP_CHANNEL_ID": "test-channel-id",
    "WS_AUTH_TOKEN": "test-ws-token",
}
for _name, _value in _DUMMY_ENV.items():
    os.environ.setdefault(_name, _value)
