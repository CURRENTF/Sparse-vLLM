import socket

import pytest

from sparsevllm.engine import model_runner
from sparsevllm.engine.model_runner import DEFAULT_MASTER_PORT, select_master_port


def test_explicit_master_port_is_preferred(monkeypatch):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    monkeypatch.setenv("SPARSEVLLM_MASTER_PORT", str(port))
    assert select_master_port() == port


def test_default_master_port_is_preferred(monkeypatch):
    monkeypatch.delenv("SPARSEVLLM_MASTER_PORT", raising=False)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    monkeypatch.setattr(model_runner, "DEFAULT_MASTER_PORT", port)
    assert select_master_port() == port


def test_explicit_occupied_master_port_fails(monkeypatch):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        monkeypatch.setenv("SPARSEVLLM_MASTER_PORT", str(port))

        with pytest.raises(RuntimeError, match=f"SPARSEVLLM_MASTER_PORT={port} is already in use"):
            select_master_port()


def test_occupied_default_master_port_uses_available_port(monkeypatch):
    monkeypatch.delenv("SPARSEVLLM_MASTER_PORT", raising=False)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", DEFAULT_MASTER_PORT))
        except OSError:
            pytest.skip(f"default port {DEFAULT_MASTER_PORT} is already occupied")

        selected = select_master_port()

    assert selected != DEFAULT_MASTER_PORT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", selected))


@pytest.mark.parametrize("value", ["", "abc", "0", "65536"])
def test_invalid_explicit_master_port_fails(monkeypatch, value):
    monkeypatch.setenv("SPARSEVLLM_MASTER_PORT", value)
    with pytest.raises(ValueError, match="SPARSEVLLM_MASTER_PORT"):
        select_master_port()
