## Official Tensormeter RTM2 Python vendor support package

This repository contains the official Python support package for the Tensormeter RTM2 from Tensor Instruments (a division of HZDR Innovation GmbH).

The source code in this repository is licensed under the Apache License 2.0. That license does not grant permission to use our company name, product names, logos, or other branding in ways that imply endorsement, affiliation, or official vendor status.

Forks and derived works are welcome under the license terms, but modified versions must not be presented as the official vendor-supported package unless they are actually distributed by Tensor Instruments.

### Repository contents


| File | Purpose |
| --- | --- |
| `rtm2.py` | Main RTM2 Python client library |
| `basic_connection_read_until.py` | Basic long-lived connection pattern: instantiate, connect, write commands, use `read_until()`, and disconnect explicitly |
| `context_manager_commands.py` | Short-script pattern using `with RTM2(...) as rtm`; demonstrates `send()`, `write()`, `.cmd.*()`, and `read_until()` |
| `threaded_reader_live_plot.py` | Threaded application-style example with `RTM2Reader`, non-blocking main loop, live plotting, state display, and interactive command input |

