# Official Tensormeter RTM2 Python vendor support package

This repository contains the official Python support package for the Tensormeter RTM2 from Tensor Instruments (a division of HZDR Innovation GmbH).

The source code in this repository is licensed under the Apache License 2.0. That license does not grant permission to use our company name, product names, logos, or other branding in ways that imply endorsement, affiliation, or official vendor status.

Forks and derived works are welcome under the license terms, but modified versions must not be presented as the official vendor-supported package unless they are actually distributed by Tensor Instruments.


## Installation

```bash
pip install git+https://github.com/hzdrinno/rtm2-python.git@v1.2.1
```

### Requirements
- Python 3.10+
- NumPy
- Matplotlib only for the live-plot example


## Quick start
```python
from rtm2 import RTM2

rtm = RTM2("169.254.178.185", 6340, timeout=0.5)

try:
    rtm.connect()

    reply = rtm.read_until("updates", send="gass", listen=0.2)
    print(rtm.get_state())

    reply = rtm.read_until("data", send="newd")
    print(reply.data)

finally:
    rtm.disconnect()
```
If the RTM2 IP or DNS name are unknown, `Discover()` can be helpful:
```python
from rtm2 import Discover

device = Discover(timeout=12.0, verbose=True)
if not devices:
    raise RuntimeError("No RTM2 broadcast received.")

for host, message in devices.items():
    print(f"Found RTM2 at {host}: {message}")
```
`Discover()` works without additional dependencies, but adapter-specific broadcast targeting uses the optional `psutil` package when available.

For a more complete walkthrough, see `examples/basic_connection_read_until.py`.

Please review the **Repository contents** section below for an overview of further examples.


## Repository contents

| File | Purpose |
| --- | --- |
| `src/rtm2/__init__.py` | Main RTM2 Python client library |
| `src/rtm2/__init__.pyi` | Generated type stub for IDE autocomplete and static analysis |
| `src/rtm2/py.typed` | Marker file for typed package distribution |
| `tools/generate_rtm2_pyi.py` | Generates the type stub from the RTM2 command registry |
| `examples/basic_connection_read_until.py` | Basic long-lived connection pattern: instantiate, connect, write commands, use `read_until()`, and disconnect explicitly |
| `examples/context_manager_commands.py` | Short-script pattern using `with RTM2(...) as rtm`; demonstrates `send()`, `write()`, `.cmd.*()`, and `read_until()` |
| `examples/threaded_reader_live_plot.py` | Threaded application-style example with `RTM2Reader`, non-blocking main loop, live plotting, state display, and interactive command input |
