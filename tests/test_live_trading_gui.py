"""Tests for the environment handling utilities in ``live_trading_gui``."""

import os

import pytest

import live_trading_gui as gui


@pytest.fixture(autouse=True)
def _reset_loader_flag():
    # Ensure each test starts with a clean loader state
    if hasattr(gui._load_env_file, "_loaded"):
        delattr(gui._load_env_file, "_loaded")
    yield
    if hasattr(gui._load_env_file, "_loaded"):
        delattr(gui._load_env_file, "_loaded")


def test_load_env_file_populates_missing_values(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("ALPACA_API_KEY=foo\nALPACA_API_SECRET=bar\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)

    gui._load_env_file()

    assert os.getenv("ALPACA_API_KEY") == "foo"
    assert os.getenv("ALPACA_API_SECRET") == "bar"


def test_load_env_file_does_not_override_existing(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("ALPACA_API_KEY=foo\nALPACA_API_SECRET=bar\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALPACA_API_KEY", "existing")
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)

    gui._load_env_file()

    assert os.getenv("ALPACA_API_KEY") == "existing"
    assert os.getenv("ALPACA_API_SECRET") == "bar"


def test_read_env_error_message_is_helpful(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)

    with pytest.raises(gui.AlpacaCredentialsError) as exc:
        gui._read_env("ALPACA_API_KEY")

    message = str(exc.value)
    assert "Missing required environment variable: ALPACA_API_KEY" in message
    assert "ALPACA_API_SECRET=your_secret_here" in message
