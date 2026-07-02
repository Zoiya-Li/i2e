"""Local <-> A800 docker box transport. Only ever targets container 29e8e3afb73f
under /home/lzy/AAAI_2026/i2e. Files move as base64 over ssh stdin (overlay fs is
not host-shared, so scp/docker cp are unavailable/forbidden)."""
from __future__ import annotations
import base64
import subprocess
import shlex

HOST = "xuhu@202.120.12.172"
PORT = "8022"
PW = "xhqweQWE123!@#"
CONTAINER = "29e8e3afb73f"
REMOTE_ROOT = "/home/lzy/AAAI_2026/i2e"
_SSH = ["sshpass", "-p", PW, "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=25", "-p", PORT, HOST]


def _exec_argv(cmd: str, interactive: bool = False) -> list[str]:
    flag = "-i " if interactive else ""
    inner = f"docker exec {flag}{CONTAINER} bash -lc {shlex.quote(cmd)}"
    return _SSH + [inner]


def run(cmd: str, timeout: int = 1800) -> str:
    """Run a shell command inside the container, return stdout."""
    p = subprocess.run(_exec_argv(cmd), capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"remote cmd failed ({p.returncode}): {p.stderr[-2000:]}")
    return p.stdout


def push(local_path: str, remote_path: str, timeout: int = 600) -> None:
    """Copy a local file into the container via base64 stdin."""
    with open(local_path, "rb") as f:
        b64 = base64.b64encode(f.read())
    cmd = f"mkdir -p $(dirname {shlex.quote(remote_path)}) && base64 -d > {shlex.quote(remote_path)}"
    p = subprocess.run(_exec_argv(cmd, interactive=True), input=b64,
                       capture_output=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"push failed: {p.stderr[-2000:]!r}")


def pull(remote_path: str, local_path: str, timeout: int = 600) -> None:
    """Copy a file out of the container to local via base64 stdout."""
    p = subprocess.run(_exec_argv(f"base64 {shlex.quote(remote_path)}"),
                       capture_output=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"pull failed: {p.stderr[-2000:]}")
    with open(local_path, "wb") as f:
        f.write(base64.b64decode(p.stdout))


def freest_gpu() -> str:
    """Index of the GPU with the most free memory (never preempt others)."""
    out = run("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits")
    best, bestfree = "0", -1
    for line in out.strip().splitlines():
        idx, free = [x.strip() for x in line.split(",")]
        if int(free) > bestfree:
            best, bestfree = idx, int(free)
    return best
