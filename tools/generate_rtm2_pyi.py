#!/usr/bin/env python3
"""
Generate src/rtm2/__init__.pyi from the RTM2 command registry.

Run from the repository root:

    python tools/generate_rtm2_pyi.py
"""

from __future__ import annotations

import argparse
import keyword
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE_DIR = SRC / "rtm2"
DEFAULT_OUTPUT = PACKAGE_DIR / "__init__.pyi"
DEFAULT_PY_TYPED = PACKAGE_DIR / "py.typed"

sys.path.insert(0, str(SRC))

from rtm2 import RTM2  # noqa: E402


_FIELD_TYPES = {
    "d": "float",
    "B": "int",
    "i": "int",
    "I": "int",
}


def _fields(fmt) -> list[str]:
    if isinstance(fmt, list):
        fmt = "".join(fmt)

    if not isinstance(fmt, str):
        return []

    return [char for char in fmt if char in _FIELD_TYPES]


def _safe_method_name(cmd: str) -> str:
    if not cmd.isidentifier() or keyword.iskeyword(cmd):
        raise ValueError(f"Cannot generate Python method for command name {cmd!r}.")
    return cmd


def _param_list(types: list[str]) -> str:
    return ", ".join(f"arg{i}: {typ}" for i, typ in enumerate(types))


def _command_stub(cmd: str, cmd_def: dict) -> list[str]:
    name = _safe_method_name(cmd)
    fmt = cmd_def["args"]
    fields = _fields(fmt)
    doc = cmd_def.get("doc", "").replace("\n", " ").strip()

    lines: list[str] = []

    if doc:
        lines.append(f"    # {doc}")

    if fmt == "":
        lines.append(f"    def {name}(self) -> None: ...")
        return lines

    if isinstance(fmt, list):
        typ = _FIELD_TYPES[fields[0]] if fields else "object"
        lines.append(f"    def {name}(self, *args: {typ}) -> None: ...")
        return lines

    if fmt == ">dd":
        lines.append("    @overload")
        lines.append(f"    def {name}(self, arg0: float) -> None: ...")
        lines.append("    @overload")
        lines.append(f"    def {name}(self, arg0: float, arg1: float) -> None: ...")
        return lines

    types = [_FIELD_TYPES[field] for field in fields]
    params = _param_list(types)

    if params:
        lines.append(f"    def {name}(self, {params}) -> None: ...")
    else:
        lines.append(f"    def {name}(self, *args: object) -> None: ...")

    return lines


def generate_stub() -> str:
    lines: list[str] = [
        '"""Type stubs for rtm2. Auto-generated; do not edit manually."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Any, overload",
        "import queue",
        "import threading",
        "import socket",
        "import numpy as np",
        "",
        "__version__: str",
        "",
        "class StateUpdate:",
        "    parameter: str",
        "    value: object",
        "    def __init__(self, parameter: str, value: object) -> None: ...",
        "",
        "class ReadResult:",
        "    updates: list[StateUpdate]",
        "    data: np.ndarray",
        "    raw_data: np.ndarray",
        "    error: None | str",
        "    def __init__(",
        "        self,",
        "        updates: list[StateUpdate],",
        "        data: np.ndarray,",
        "        raw_data: np.ndarray,",
        "        error: None | str = None,",
        "    ) -> None: ...",
        "",
        "class PacketFramingError(Exception): ...",
        "",
        "class _CmdFacade:",
    ]

    for cmd, cmd_def in RTM2._COMMANDS.items():
        lines.extend(_command_stub(cmd, cmd_def))

    lines.extend(
        [
            "",
            "def SwitState(DRVn: list, DRVp: list, SNSn: list, SNSp: list) -> int: ...",
            "def Discover(",
            "    timeout: float = 12.0,",
            "    primer_addr: str | None = None,",
            "    primer_port: int = 61556,",
            ") -> tuple[str, str] | None: ...",
            "",
            "class RTM2:",
            "    host: str",
            "    port: int",
            "    tcp: socket.socket | None",
            "    cmd: _CmdFacade",
            "",
            "    def __init__(self, host: str, port: int, timeout: float = 1.0) -> None: ...",
            "    def __enter__(self) -> RTM2: ...",
            "    def __exit__(self, exc_type: Any, exc_val: Any, traceback: Any) -> None: ...",
            "    def connect(self) -> None: ...",
            "    def disconnect(self) -> None: ...",
            "    def get_state(self) -> dict[str, object]: ...",
            "    def send(self, cmd: str, *pars: object) -> None: ...",
            "    def write(self, usrstr: str) -> None: ...",
            "    def read(self, max_packets: int = 100) -> ReadResult: ...",
            "    def read_until(",
            "        self,",
            "        *terms: str,",
            "        timeout: float = 10.0,",
            "        listen: float = 0.0,",
            "        send: str | tuple[Any, ...] | list[Any] | None = None,",
            "    ) -> ReadResult: ...",
            "",
            "class RTM2Reader:",
            "    rtm: RTM2",
            "    results: queue.Queue[Any]",
            "    def __init__(self, rtm: RTM2) -> None: ...",
            "    def start(self) -> None: ...",
            "    def stop(self, timeout: float | None = 2.0) -> None: ...",
            "    def _run(self) -> None: ...",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output stub path. Default: src/rtm2/__init__.pyi",
    )
    parser.add_argument(
        "--no-py-typed",
        action="store_true",
        help="Do not create/touch src/rtm2/py.typed.",
    )
    args = parser.parse_args()

    if not PACKAGE_DIR.exists():
        raise FileNotFoundError(f"Expected package directory does not exist: {PACKAGE_DIR}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate_stub(), encoding="utf-8")
    print(f"Wrote {args.output}")

    if not args.no_py_typed:
        DEFAULT_PY_TYPED.touch()
        print(f"Touched {DEFAULT_PY_TYPED}")


if __name__ == "__main__":
    main()