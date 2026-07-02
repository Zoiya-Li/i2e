"""Unit tests for remote transport helper — command construction only (no live SSH)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_remote_root_is_workspace():
    from work.remote import REMOTE_ROOT
    assert REMOTE_ROOT == "/home/lzy/AAAI_2026/i2e"


def test_exec_argv_targets_only_allowed_container():
    from work.remote import _exec_argv
    argv = _exec_argv("echo hi")
    joined = " ".join(argv)
    assert "29e8e3afb73f" in joined
    assert "docker exec" in joined
    assert "-p 8022" in joined
    assert "xuhu@202.120.12.172" in joined


def test_exec_argv_interactive_flag():
    from work.remote import _exec_argv
    argv = _exec_argv("cat", interactive=True)
    joined = " ".join(argv)
    assert "docker exec -i " in joined


def test_push_argv_uses_interactive():
    from work.remote import _exec_argv
    argv = _exec_argv("base64 -d > /tmp/x", interactive=True)
    joined = " ".join(argv)
    assert "docker exec -i " in joined
