"""
RTM2 Python Client Library

Requires: Python 3.10+

Users should import the `RTM2` class.
The main interaction happens via asynchronous writes and polling reads.

Schematic example:

```
from rtm2 import RTM2

rtm = RTM2("192.168.1.10", 6340)
rtm.connect()

while measuring:
    rtm.write("newd")
    result = rtm.read()

rtm.disconnect()
```

The library also contains helper functions intended for users:

- `SwitState([], [], [], [])` formats simple port function lists into switch matrix state integers.
- `Discover()` can intercept IPv4 UDP broadcast messages from available RTM2 devices that do not have a live TCP connection.

`RTM2Reader` sets up a fully asynchronous reader thread. If this is used, users should not call
`read()` or `read_until()` on the same `RTM2` instance anymore, but instead consume `RTM2Reader.results`.
"""

import socket
import struct
import select
import time
import numpy as np
from dataclasses import dataclass
import threading    # for the optional RTM2Reader thread
import queue        # for the optional RTM2Reader thread output


__version__ = "1.2.0"


@dataclass
class StateUpdate:
    parameter: str
    value: object


@dataclass
class ReadResult:
    """
    Return value of `read()` and `read_until()`.

    - `updates` contains incremental setting updates.
    - `data` is a 2D NumPy array, rows representing time.
    - `raw_data` is a 2D NumPy array, rows representing time.
    - `error` is `None` or a string explaining a protocol-level error.
    """
    updates: list[StateUpdate]
    data: np.ndarray
    raw_data: np.ndarray
    error: None | str = None


class PacketFramingError(Exception):
    """Raised internally when packet framing or structural packet integrity is lost."""
    pass


class _CmdFacade:
    """Namespace populated with generated command methods by `RTM2._build_cmd_facade()`."""
    pass


# --- Generic command payload encoding/decoding helpers ---

_STRUCT_CASTS = {
    "d": float,
    "B": int,
    "i": int,
    "I": int,
}


def _encode_payload(fmt, args: tuple) -> bytes:
    """
    Encode a command payload from a compact declarative format.

    Supported formats:
        ""          -> empty payload, requires no args
        ">d", ">i"  -> regular struct formats
        ">dd"       -> two doubles; if one arg is supplied, the second becomes 0.0
        [">I"]      -> counted sequence; count is sent as >i followed by all values
    """
    if fmt == "":
        if args:
            raise ValueError(f"Expected 0 arguments, got {len(args)}.")
        return b""

    if isinstance(fmt, list):
        field = [char for char in fmt[0] if char in _STRUCT_CASTS][0]
        values = [_STRUCT_CASTS[field](arg) for arg in args]
        count = len(values)
        return struct.pack(">i", count) + struct.pack(f">{count}{field}", *values)

    fields = [char for char in fmt if char in _STRUCT_CASTS]

    # RTM2 rampable setpoints are encoded as two doubles. The second argument is
    # optional for user convenience and defaults to 0.0, matching firmware behavior.
    if fmt == ">dd" and len(args) == 1:
        args = (args[0], 0.0)

    if len(args) != len(fields):
        raise ValueError(f"Expected {len(fields)} arguments for {fmt}, got {len(args)}.")

    values = [_STRUCT_CASTS[field](arg) for field, arg in zip(fields, args)]
    return struct.pack(fmt, *values)


def _decode_payload(fmt, payload: bytes):
    """
    Decode a reply payload from a compact declarative format.

    Supported formats:
        ""          -> acknowledged, but no cacheable state value; returns None
        ">d", ">i"  -> regular struct formats
        [">I"]      -> counted sequence; returns a tuple
        "data"      -> rows/cols header followed by big-endian doubles; returns a 2D NumPy array
    """
    if fmt == "":
        return None

    if fmt == "data":
        if len(payload) < 8:
            raise ValueError("Data payload is too short to contain rows/cols header.")

        rows, cols = struct.unpack(">ii", payload[:8])
        expected_len = 8 + rows * cols * 8
        if len(payload) != expected_len:
            raise ValueError(f"Data payload length mismatch: expected {expected_len}, got {len(payload)}.")

        return np.frombuffer(payload[8:], dtype=">d").reshape((rows, cols))

    if isinstance(fmt, list):
        field = [char for char in fmt[0] if char in _STRUCT_CASTS][0]
        item_fmt = fmt[0]

        if len(payload) < 4:
            raise ValueError("Sequence payload is too short to contain a count field.")

        count = struct.unpack(">i", payload[:4])[0]
        expected_len = 4 + count * struct.calcsize(item_fmt)
        if len(payload) != expected_len:
            raise ValueError(f"Sequence payload length mismatch: expected {expected_len}, got {len(payload)}.")

        return struct.unpack(f">{count}{field}", payload[4:])

    expected_len = struct.calcsize(fmt)
    if len(payload) != expected_len:
        raise ValueError(f"Struct payload length mismatch for {fmt}: expected {expected_len}, got {len(payload)}.")

    values = struct.unpack(fmt, payload)
    return values[0] if len(values) == 1 else values


# --- User helper functions ---

def SwitState(DRVn: list, DRVp: list, SNSn: list, SNSp: list) -> int:
    """
    Accept four lists of BNC port numbers and return one switch matrix state integer.

    Each list defines which BNC ports are connected to DRV-, DRV+, SNS-, SNS+.
    The resulting integer can be passed to `swit`.
    """
    result = 0

    def assign_port(n: int, offset: int):
        nonlocal result
        if isinstance(n, int) and 1 <= n <= 8:
            result |= (1 << (n - 1 + offset))
        else:
            raise ValueError(f"Invalid switch port: {n!r}. Expected integer 1..8.")

    for n in DRVn:
        assign_port(n, 0)
    for n in DRVp:
        assign_port(n, 8)
    for n in SNSn:
        assign_port(n, 16)
    for n in SNSp:
        assign_port(n, 24)

    return result


def Discover(
    timeout: float = 12.0,
    primer_addrs: list[str] | None = None,
    primer_port: int = 61556,
    verbose: bool = False,
) -> dict[str, str]:
    """
    Listen for an IPv4 UDP broadcast announcement from an RTM2 device.

    Discovery listens on UDP port 61557 and sends short primer packets to
    likely IPv4 broadcast addresses first. If `psutil` is installed, those
    addresses are derived from the client's active IPv4 network adapters.
    Without `psutil`, the function falls back to a small set of generic
    broadcast addresses.

    If `primer_addrs` is supplied, those addresses are tried in addition to
    the automatic and fallback broadcast targets.

    If `verbose` is True, the primer targets are printed before sending.

    Returns a dict mapping sender_ip to announcement message. The dict is
    empty if no RTM2 announcement is received before the timeout expires.
    """

    # Put any user-supplied primer_addrs into the targets set.
    targets: set[str] = set(primer_addrs or [])

    # psutil is optional. It gives a clean cross-platform way to inspect
    # local adapters. Without it, discovery still works via fallback targets.
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        psutil = None
        import warnings

        warnings.warn(
            "RTM2 Discover: optional package 'psutil' is not installed; "
            "falling back to generic broadcast targets only. Install psutil "
            "for better adapter-specific discovery:\n"
            "  > pip install psutil",
            UserWarning,
            stacklevel=2,
        )

    if psutil is not None:
        for _if_name, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                # This discovery method sends IPv4 UDP broadcasts only.
                # Ignore IPv6, MAC/link-layer, and other non-IPv4 adapter records.
                if addr.family != socket.AF_INET:
                    continue

                ip = getattr(addr, "address", None)
                netmask = getattr(addr, "netmask", None)

                if not ip or ip.startswith("127."):
                    continue

                # Calculate the directed broadcast address from address and netmask.
                # Example: 192.168.178.34 / 255.255.255.0 -> 192.168.178.255
                if netmask:
                    try:
                        ip_parts = [int(part) for part in ip.split(".")]
                        mask_parts = [int(part) for part in netmask.split(".")]
                        if len(ip_parts) != 4 or len(mask_parts) != 4:
                            raise ValueError
                        if any(part < 0 or part > 255 for part in ip_parts + mask_parts):
                            raise ValueError

                        broadcast_parts = [
                            (ip_part & mask_part) | (~mask_part & 0xFF)
                            for ip_part, mask_part in zip(ip_parts, mask_parts)
                        ]
                        broadcast = ".".join(str(part) for part in broadcast_parts)
                        if broadcast != ip:
                            targets.add(broadcast)
                    except (TypeError, ValueError):
                        pass

    # Generic fallbacks. These are cheap and cover common cases even when
    # adapter-derived broadcast targets are incomplete or unavailable.
    targets.add("169.254.255.255")
    targets.add("192.168.255.255")
    targets.add("255.255.255.255")

    if verbose:
        print("RTM2 Discover primer targets:")
        for addr in sorted(targets):
            print(f"  {addr}:{primer_port}")

    deadline = time.monotonic() + timeout
    found: dict[str, str] = {}

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # RTM2 always broadcasts to port 61557
        sock.bind(("", 61557))

        # Send all primers first, then listen until the total timeout expires.
        # We use a separate target-side destination port (`primer_port`), while
        # using port 61557 to send from and listen for the RTM's broadcasts.
        for addr in targets:
            try:
                sock.sendto(b"UDP broadcast receive primer", (addr, primer_port))
            except OSError:
                # Some broadcast addresses may be rejected by the OS depending
                # on adapter state, route table, VPNs, or firewall policy.
                # Ignore those and keep trying the remaining targets.
                pass

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return found

            sock.settimeout(remaining)

            try:
                payload, sender = sock.recvfrom(1024)
            except socket.timeout:
                return found
            except ConnectionResetError:
                # Windows may raise this for UDP sockets after an earlier
                # primer hit an unreachable target/port. Ignore it and keep
                # listening until the discovery timeout expires.
                continue

            if b"RTM2" in payload:
                found[sender[0]] = payload.decode("ascii", errors="replace")



class RTM2:
    """
    Per-device TCP client for Tensormeter RTM2.

    Parameters
    --------
    host:
        IP address or DNS name of the RTM2.
    port:
        TCP port of the RTM2. The default is 6340.
    timeout:
        TCP connection timeout and initial socket read timeout for this
        session. For normal use, choose this once when creating the RTM2
        object. Advanced users may adjust the live socket timeout directly
        via the instance's `.tcp.settimeout(...)` after connecting.

    `send()`, `write()`, and generated `cmd.*()` methods validate command names and
    arguments and raise exceptions on malformed user input.

    `read()` returns protocol-level issues in `ReadResult.error`, because it can still
    return partial useful contents from the same receive pass. Transport failures raise.
    """

    _COMMANDS = {
        # Doubles
        "avgt": {"args":  ">d",   "reply": ">d",   "type": "state",    "doc": "Set Averaging Time."},
        "cpro": {"args":  ">d",   "reply": ">d",   "type": "state",    "doc": "Set Current Limit."},
        "ipro": {"args":  ">d",   "reply": ">d",   "type": "state",    "doc": "Set Current Limit."},
        "vpro": {"args":  ">d",   "reply": ">d",   "type": "state",    "doc": "Set Output Voltage Limit."},
        "lfrq": {"args":  ">d",   "reply": ">d",   "type": "state",    "doc": "Set AC Frequency."},
        "sres": {"args":  ">d",   "reply": ">d",   "type": "state",    "doc": "Set Series Resistance. Negative values enable auto-selection."},
        "crng": {"args":  ">d",   "reply": ">d",   "type": "state",    "doc": "Set Current measurement range. Negative values enable auto-selection."},
        "vorg": {"args":  ">d",   "reply": ">d",   "type": "state",    "doc": "Set Output Voltage measurement range. Negative values enable auto-selection."},
        "virg": {"args":  ">d",   "reply": ">d",   "type": "state",    "doc": "Set Input Voltage measurement range. Negative values enable auto-selection."},
        "phsh": {"args":  ">d",   "reply": ">d",   "type": "state",    "doc": "Set AC Phase Shift from Reference Input."},
        "time": {"args":  ">d",   "reply": ">d",   "type": "state",    "doc": "Request the timestamp difference between the provided timestamp and the RTM2's internal one."},

        # Unsigned bytes
        "cmod": {"args":  ">B",   "reply": ">B",   "type": "state",    "doc": "Set Output Control Mode."},
        "wfmd": {"args":  ">B",   "reply": ">B",   "type": "state",    "doc": "Set Waveform Mode."},
        "modq": {"args":  ">B",   "reply": ">B",   "type": "state",    "doc": "Detected Analysis Mode."},
        "amod": {"args":  ">B",   "reply": ">B",   "type": "state",    "doc": "Set Analysis Mode."},
        "mult": {"args":  ">B",   "reply": ">B",   "type": "state",    "doc": "Set Multisample Mode."},
        "refm": {"args":  ">B",   "reply": ">B",   "type": "state",    "doc": "Set Reference Multiplexer Input."},
        "phlk": {"args":  ">B",   "reply": ">B",   "type": "state",    "doc": "Set Phase Locking Behavior."},
        "snsa": {"args":  ">B",   "reply": ">B",   "type": "state",    "doc": "Set SNS preamplifier Mode."},
        "coax": {"args":  ">B",   "reply": ">B",   "type": "state",    "doc": "Set BNC Coax Mode."},
        "drvp": {"args":  ">B",   "reply": ">B",   "type": "state",    "doc": "Set Drive Polarity Mode."},

        # Integers
        "meas": {"args":  ">i",   "reply": ">i",   "type": "state",    "doc": "Set Data sample counter. Negative values enable infinite sampling."},

        # Parameter-less acknowledgment commands
        "trig": {"args":  "",     "reply": "",     "type": "state",    "doc": "Begin a new demodulation window immediately."},
        "puls": {"args":  "",     "reply": "",     "type": "state",    "doc": "Begin the pulse train (or arbitrary waveform) generation."},
        "gass": {"args":  "",     "reply": "",     "type": "state",    "doc": "Request all device settings."},
        "cldt": {"args":  "",     "reply": "",     "type": "state",    "doc": "Clear device side data buffer."},
        "srup": {"args":  "",     "reply": "",     "type": "state",    "doc": "Series Resistance Up."},
        "srdn": {"args":  "",     "reply": "",     "type": "state",    "doc": "Series Resistance Down."},
        "crup": {"args":  "",     "reply": "",     "type": "state",    "doc": "Current measurement range Up."},
        "crdn": {"args":  "",     "reply": "",     "type": "state",    "doc": "Current measurement range Down."},
        "voru": {"args":  "",     "reply": "",     "type": "state",    "doc": "Voltage Output measurement range Up."},
        "vord": {"args":  "",     "reply": "",     "type": "state",    "doc": "Voltage Output measurement range Down."},
        "viru": {"args":  "",     "reply": "",     "type": "state",    "doc": "Voltage Input measurement range Up."},
        "vird": {"args":  "",     "reply": "",     "type": "state",    "doc": "Voltage Input measurement range Down."},

        # Rampable parameters: 1 or 2 user args, encoded as 2 doubles, replies as 1 double
        "camp": {"args":  ">dd",  "reply": ">d",   "type": "state",    "doc": "Set Current Amplitude setpoint. Optional 2nd argument: Time to arrival."},
        "cudc": {"args":  ">dd",  "reply": ">d",   "type": "state",    "doc": "Set Current DC setpoint. Optional 2nd argument: Time to arrival."},
        "vamp": {"args":  ">dd",  "reply": ">d",   "type": "state",    "doc": "Set Voltage Amplitude setpoint. Optional 2nd argument: Time to arrival."},
        "vodc": {"args":  ">dd",  "reply": ">d",   "type": "state",    "doc": "Set Voltage DC setpoint. Optional 2nd argument: Time to arrival."},

        # Special structured commands
        "dio0": {"args":  ">Bd",  "reply": ">Bd",  "type": "state",    "doc": "Set DIO0 mode."},
        "dio1": {"args":  ">Bd",  "reply": ">Bd",  "type": "state",    "doc": "Set DIO1 mode."},
        "swit": {"args": [">I"],  "reply": [">I"], "type": "state",    "doc": "Define Switch Matrix states."},
        "selc": {"args": [">i"],  "reply": [">i"], "type": "state",    "doc": "Set indices of data channels that will be sent as reply to `newd` calls."},
        "puar": {"args": [">d"],  "reply": [">d"], "type": "state",    "doc": "Set pulse parameter array entries."},

        # Data commands
        "newd": {"args":  "",     "reply": "data", "type": "data",     "doc": "Request all new data rows, i.e. previously unsent rows."},
        "alld": {"args":  "",     "reply": "data", "type": "data",     "doc": "Request all data rows."},
        "rawd": {"args":  ">i",   "reply": "data", "type": "raw_data", "doc": "Request a number of rows of raw ADC samples."},
    }

    def __init__(self, host: str, port: int, timeout: float = 1.0):
        if timeout < 0:
            raise ValueError("timeout must be non-negative.")

        self.host = host
        self.port = port
        self._timeout = float(timeout)
        self.tcp: None | socket.socket = None
        self._is_connected = False
        self._state: dict[str, object] = {}
        self.cmd = self._build_cmd_facade()

    def _build_cmd_facade(self):
        cmd_obj = _CmdFacade()

        for cmd, cmd_def in self._COMMANDS.items():
            def make_method(cmd):
                def method(*args):
                    return self.send(cmd, *args)
                return method

            method = make_method(cmd)
            method.__name__ = cmd
            method.__doc__ = cmd_def.get("doc", f"Auto-generated wrapper for {cmd!r}.")
            setattr(cmd_obj, cmd, method)

        return cmd_obj

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self.disconnect()

    def connect(self):
        """Open the TCP connection. Hostnames may resolve to IPv4 or IPv6 addresses."""
        if self._is_connected:
            return

        try:
            self.tcp = socket.create_connection((self.host, self.port), timeout=self._timeout)
            self._is_connected = True
        except OSError as e:
            self.tcp = None
            self._is_connected = False
            raise ConnectionError(f"Failed to connect to RTM2 at {self.host}:{self.port}: {e}") from e

    def disconnect(self):
        """Close the TCP connection. Disconnect is best-effort and does not raise."""
        sock = self.tcp
        self.tcp = None
        self._is_connected = False

        if not sock:
            return

        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

        try:
            sock.close()
        except OSError:
            pass

    def get_state(self) -> dict[str, object]:
        return self._state.copy()

    def send(self, cmd: str, *pars):
        """
        Send one command plus arguments to the RTM2.

        Raises ValueError for unknown commands or malformed arguments.
        Raises ConnectionError for transport-level failures.
        """
        if not self._is_connected or self.tcp is None:
            raise ConnectionError("Cannot write to device: Not connected.")

        cmd_def = self._COMMANDS.get(cmd)
        if not cmd_def:
            raise ValueError(f"Unknown RTM2 command: {cmd!r}")

        try:
            payload = _encode_payload(cmd_def["args"], pars)
        except (ValueError, IndexError, struct.error) as e:
            raise ValueError(f"Invalid parameters for RTM2 command {cmd!r}: {pars}") from e

        packet_size = struct.pack(">i", len(payload) + 4)
        packet = packet_size + cmd.encode("ascii") + payload

        try:
            self.tcp.sendall(packet)
        except OSError as e:
            self._is_connected = False
            raise ConnectionError(f"Socket error while sending {cmd!r}: {e}") from e

    def write(self, usrstr: str):
        """Parse a user-style command string and forward it to `send()`."""
        parts = usrstr.split()
        if not parts:
            raise ValueError("No RTM2 command provided.")
        self.send(*parts)

    def _recv_exact(self, count: int) -> bytes | None:
        """
        Receive exactly `count` bytes using the socket timeout.

        Returns None when the socket times out before all bytes are received.
        Raises ConnectionError if the peer closes the socket.
        """
        buf = bytearray()

        while len(buf) < count:
            try:
                chunk = self.tcp.recv(count - len(buf))
            except socket.timeout:
                return None

            if not chunk:
                raise ConnectionError("Socket connection closed while receiving data.")

            buf.extend(chunk)

        return bytes(buf)

    def _flush_rx_buffer(self, grace_timeout: float = 0.1):
        """Best-effort receive-buffer drain after packet framing loss."""
        deadline = time.monotonic() + grace_timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            readable, _, _ = select.select([self.tcp], [], [], remaining)
            if not readable:
                break

            chunk = self.tcp.recv(4096)
            if not chunk:
                raise ConnectionError("Socket connection closed while flushing receive buffer.")

    def _read_one_packet(self):
        """
        Read one complete RTM2 reply packet.

        Returns None if no header appears within the current socket timeout.
        Raises PacketFramingError if a packet begins but cannot be completed.
        """
        header = self._recv_exact(4)
        if header is None:
            return None

        payload_size = struct.unpack(">i", header)[0] - 4
        if payload_size < 0:
            raise PacketFramingError(f"Invalid packet payload size: {payload_size}")

        cmd_bytes = self._recv_exact(4)
        if cmd_bytes is None:
            raise PacketFramingError("Timed out while receiving packet command")

        payload = self._recv_exact(payload_size)
        if payload is None:
            raise PacketFramingError("Timed out while receiving packet payload")

        return cmd_bytes.decode("ascii", errors="replace"), payload

    def _parse_packet(self, cmd, payload):
        cmd_def = self._COMMANDS.get(cmd)

        if not cmd_def:
            return None, None, f"Unknown incoming command: {cmd}"

        packet_type = cmd_def["type"]

        try:
            content = _decode_payload(cmd_def["reply"], payload)
        except Exception as e:
            if packet_type in {"data", "raw_data"}:
                raise PacketFramingError(f"Malformed {packet_type} packet for command {cmd}") from e
            return None, None, f"Decode error for command {cmd}"

        if packet_type == "state":
            return "state", StateUpdate(cmd, content), None

        if packet_type in {"data", "raw_data"}:
            return packet_type, content, None

        return None, None, f"Unsupported packet type: {packet_type}"

    def read(self, max_packets: int = 100) -> ReadResult:
        """
        Read and drain available RTM2 replies.

        The first packet waits up to `self._timeout` seconds. After each received
        packet, `select()` with timeout 0.0 decides whether another already-
        available packet should be drained. If so, it is retrieved using the 
        regular `self._timeout`.

        Protocol-level parse/framing problems are returned in `ReadResult.error`.
        Transport-level failures raise ConnectionError.
        """
        if not self._is_connected or self.tcp is None:
            raise ConnectionError("Cannot read from device: Not connected.")

        packets = 0
        updates = []
        data = []
        raw_data = []
        error = None

        while packets < max_packets:
            try:
                packet = self._read_one_packet()
                if packet is None:
                    break

                packet_type, content, parse_error = self._parse_packet(*packet)
                if error is None:
                    error = parse_error

                if packet_type == "state":
                    updates.append(content)
                    if content.value is not None:
                        self._state[content.parameter] = content.value
                elif packet_type == "data":
                    if content.size:
                        data.append(content)
                elif packet_type == "raw_data":
                    if content.size:
                        raw_data.append(content)

                packets += 1

                # After one full packet, only continue if another packet is already readable.
                # This replaces the previous self.tcp.settimeout(0.0) drain trick.
                readable, _, _ = select.select([self.tcp], [], [], 0.0)
                if not readable:
                    break

            except PacketFramingError as e:
                frame_error = f"{e} -> Packet framing lost. Flushing receive buffer."
                error = f"{error} Additionally: {frame_error}" if error else frame_error
                try:
                    self._flush_rx_buffer()
                except OSError as flush_error:
                    self._is_connected = False
                    raise ConnectionError(f"Socket error while flushing receive buffer: {flush_error}") from flush_error
                break

            except OSError as e:
                self._is_connected = False
                raise ConnectionError(f"Socket error while reading: {e}") from e

        return ReadResult(
            updates=updates,
            data=np.concatenate(data).astype(np.float64) if data else np.empty((0, 0)),
            raw_data=np.concatenate(raw_data).astype(np.float64) if raw_data else np.empty((0, 0)),
            error=error,
        )

    def read_until(self, *terms: str, timeout: float = 10.0, listen: float = 0.0, send=None) -> ReadResult:
        """
        Repeatedly call `read()` until selected reply content appears and the
        minimum listen time has passed, an error occurs, or the outer timeout expires.
        """
        if not self._is_connected:
            raise ConnectionError("Cannot read from device: Not connected.")

        if timeout < 0:
            raise ValueError("timeout must be non-negative.")
        if listen < 0:
            raise ValueError("listen must be non-negative.")

        if send is not None:
            if isinstance(send, str):
                self.write(send)
            elif isinstance(send, (tuple, list)):
                if not send:
                    raise ValueError("send= tuple/list must contain at least a command name.")
                self.send(*send)
            else:
                raise TypeError("send= must be None, a command string, or a tuple/list for self.send().")

        wanted = {str(term).lower() for term in terms} if terms else {"any"}
        component_terms = {"any", "update", "updates", "data", "raw", "raw_data", "error"}
        wanted_components = wanted & component_terms
        wanted_parameters = wanted - component_terms

        def result_matches(result: ReadResult) -> bool:
            if result.error:
                return True
            if "any" in wanted_components:
                return bool(result.updates or result.data.size or result.raw_data.size)
            if {"update", "updates"} & wanted_components and result.updates:
                return True
            if "data" in wanted_components and result.data.size:
                return True
            if {"raw", "raw_data"} & wanted_components and result.raw_data.size:
                return True
            if "error" in wanted_components and result.error:
                return True
            if wanted_parameters:
                return any(upd.parameter in wanted_parameters for upd in result.updates)
            return False

        updates = []
        data = []
        raw_data = []
        error = None
        matched = False

        start = time.monotonic()
        timeout_deadline = start + timeout
        listen_deadline = start + listen

        while time.monotonic() <= timeout_deadline:
            result = self.read()

            if result.updates:
                updates.extend(result.updates)
            if result.data.size:
                data.append(result.data)
            if result.raw_data.size:
                raw_data.append(result.raw_data)
            if error is None and result.error:
                error = result.error

            if result_matches(result):
                matched = True
            if result.error:
                break
            if matched and time.monotonic() >= listen_deadline:
                break

        return ReadResult(
            updates=updates,
            data=np.concatenate(data).astype(np.float64) if data else np.empty((0, 0)),
            raw_data=np.concatenate(raw_data).astype(np.float64) if raw_data else np.empty((0, 0)),
            error=error,
        )


class RTM2Reader:
    """
    Owns one background thread that repeatedly calls `RTM2.read()`
    and forwards non-empty results into a queue.
    """

    def __init__(self, rtm: RTM2):
        self.rtm = rtm
        self.results: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="rtm2-reader", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 2.0):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                result = self.rtm.read()
                if result.updates or result.data.size or result.raw_data.size or result.error:
                    self.results.put(result)
            except Exception as exc:
                self.results.put(exc)
                break
