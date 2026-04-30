import socket
import struct
import logging
import time
import numpy as np
from dataclasses import dataclass
import threading    # for the optional RTM2Reader thread
import queue        # for the optional RTM2Reader thread output


__version__ = "1.1.0"

"""
RTM2 Python Client Library

Requires: Python 3.9+

Users should import the `RTM2` class.
The main interaction happens via non-blocking writes and polling reads of that class.

Schematic example:

```
from rtm2 import RTM2

rtm = RTM2("192.168.1.10", 6340)
rtm.connect()

# measurement loop
while measuring:
    rtm.write("newd")
    result = rtm.read()

rtm.disconnect()
```

The library also contains helper functions intended for users:

- `SwitState([], [], [], [])` allows formatting simple port function lists into switch matrix state integers
- `Discover()` can be called to intercept UDP broadcast messages from available RTMs that don't have a live TCP connection

The `RTM2Reader` class sets up a fully asynchronous reader thread. If this is used, users should not use the `read()` or
`read_until()` function of the main `RTM2` class anymore, but instead consume the output queue of `RTM2Reader`.
"""


logger = logging.getLogger(__name__)


@dataclass
class StateUpdate:
    parameter: str
    value: object


@dataclass
class ReadResult:
    """
    This is returned by the `read()` function. Generally, all of the four components
    mentioned below will be empty (or `None` in the case of `error`), unless one or
    several specific requests were issued before calling `read()`.

    - `read().updates` contains a list of incremental setting updates.
    - `read().data` is a 2D NumPy array, rows representing time.
    - `read().raw_data` is a 2D NumPy array, rows representing time.
    - `read().error` is `None` or a string explaining the error
    """
    updates: list[StateUpdate]
    data: np.ndarray
    raw_data: np.ndarray
    error: None | str = None


class PacketFramingError(Exception):
    """Raised when packet framing or structural packet integrity is lost."""
    pass


class _CmdFacade:
    """
    Empty namespace to be used in the method auto-generator. Check `RTM2._build_cmd_facade()`
    for more details.
    """
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

    # Regular struct payload.
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


# --- Other helpers meant for user interaction ---

def SwitState(DRVn: list, DRVp: list, SNSn: list, SNSp: list) -> int:
    """
    This function accepts four lists of numbers and outputs a single int number.
    Each list defines which BNC ports are to be connected to DRV-, DRV+, SNS-, SNS+, respectively.
    The resulting number can be directly used as argument for a `write('swit ...')` call.

    Schematic example, to set up a Zero-Offset-Hall configuration:

    ```
    sw1 = SwitState([1], [3], [2], [4])
    sw2 = SwitState([4], [2], [1], [3])
    MyRTM.write(f'swit {sw1} {sw2}')
    ```
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
    primer_addr: str | None = None,
    primer_port: int = 61556,
) -> tuple[str, str] | None:
    """
    Wait for a single UDP broadcast announcement from an RTM2 device on port 61557.
    Sends a UDP primer broadcast first, using `primer_port` (61556 by default).
    Only packet containing `b'RTM2'` will be considered.

    If no `primer_addr` is provided, a few common broadcast addresses are tried
    first as a best-effort compatibility measure.

    Returns (msg, sender_ip) or None on timeout.
    """
    fallback_primers = [
        "169.254.255.255", # link-local, where RTM2 IPs are sometimes difficult to find
        "192.168.255.255", # typical LAN environment
        "255.255.255.255"  # last-ditch effort, try a generic primer, which will be probably blocked
    ]

    targets = [primer_addr] if primer_addr else fallback_primers

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(timeout)
        sock.bind(("", 61557))
        
        for addr in targets:
            sock.sendto(b"UDP broadcast receive primer", (addr, primer_port))
        logger.info(f"Sent UDP broadcast receive primer. Now waiting for a UDP broadcast for {timeout} s...")

        try:
            payload, sender = sock.recvfrom(1024)
            return (payload.decode("ascii"), sender[0]) if b"RTM2" in payload else None
        except socket.timeout:
            return None


# --- The main class ---

class RTM2:
    """
    This class abstracts communication on a per-device level.
    
    It holds a `_state` property, which is a dictionary of the most recent
    actually known device settings. This dictionary is automatically updated
    each time the `read()` function is called based on the `StateUpdates` it
    returns. Can be accessed via the `get_state()` function.

    Commands that acknowledge receipt without returning a value are included in
    `ReadResult.updates` with value `None`, but are not cached in `_state`.

    The class also holds the TCP connection settings. The default timeout is
    1.0 seconds. If a snappier timeout is preferred, it should be supplied
    when the class is being called first (and initialized). Timeout can be
    adjusted at any time by overwriting the `.timeout` property.

    Writing commands to the RTM can be done in three different ways: `send()`
    provides the baseline functionality, accepting a command name (`str`) followed
    by parameters. `write` is a small wrapper, accepting a single string argument
    containing both the command and parameters. The third way are distinct methods
    per command, e.g. `RTM2.cmd.vodc()` that take only the numeric arguments.

    While all of the writing/commanding functions are fire-and-forget functions,
    `read()` is an independent polling function that will always attempt to deplete
    the entire TCP buffer. It will only return after finding at least one full RTM2
    reply in the TCP buffer or after TCP timeout.

    `read_until()` is a blocking convenience wrapper around `read()`.

    Generally, a continuous TCP connection is preferred. Context-manager
    support is provided and is useful for e.g. one-time configuration bursts. 
    """
    
    # --- Central command registry ---
    #
    # Each command defines:
    #   - args: outgoing payload format
    #   - reply: incoming payload format
    #   - type: state, data, or raw_data
    #   - doc: docstring for autogenerated wrapper functions
    #
    # Format entries:
    #   ""       -> empty payload / acknowledgment
    #   ">d"     -> regular struct format
    #   ">dd"    -> two doubles; one user argument is accepted and padded with 0.0
    #   [">I"]   -> counted sequence, encoded as count (>i) followed by values
    #   "data"   -> rows/cols header followed by big-endian doubles


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
        "modq": {"args":  ">B",   "reply": ">B",   "type": "state",    "doc": "Detected Analysis Mode."},  # not meant to be sent, only received
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
        self.host = host
        self.port = port
        self.timeout = timeout
        self.tcp: None | socket.socket = None  # Socket is created during connect()
        self._is_connected = False
        self._state: dict[str, object] = {}
        self.cmd = self._build_cmd_facade()

    def _build_cmd_facade(self):
        """
        Create and populate the `cmd` namespace.
        
        For each entry in the command registry (`RTM2._COMMANDS`), a method is
        generated that forwards its arguments to `RTM2.send()`.
        """
        cmd_obj = _CmdFacade()

        for cmd, cmd_def in self._COMMANDS.items():

            def make_method(cmd):
                def method(*args):
                    return self.send(cmd, *args)
                return method

            m = make_method(cmd)
            m.__name__ = cmd
            m.__doc__ = cmd_def.get("doc", f"Auto-generated wrapper for {cmd!r}.")
            setattr(cmd_obj, cmd, m)

        return cmd_obj

    # --- Context Manager ---
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self.disconnect()

    # --- Connection ---
    def connect(self):
        try:
            self.tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp.settimeout(self.timeout)
            self.tcp.connect((self.host, self.port))
            self._is_connected = True
            logger.info(f"Connected to RTM2 at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    def disconnect(self):
        if self.tcp and self._is_connected:
            try:
                self.tcp.shutdown(socket.SHUT_RDWR)
                self.tcp.close()
                self._is_connected = False
                logger.info("Disconnected securely.")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

    def get_state(self) -> dict[str, object]:
        return self._state.copy()

    # --- The main writer ---

    def send(self, cmd: str, *pars):
        """
        Core transport function for sending one command plus arguments to the RTM2.

        Malformed commands will log a warning and be ignored, but won't raise Exceptions.
        """
        if not self._is_connected:
            raise ConnectionError("Cannot write to device: Not connected.")

        try:
            cmd_def = self._COMMANDS.get(cmd)
            if not cmd_def:
                logger.warning(f"Unknown command: {cmd}")
                return

            payload = _encode_payload(cmd_def["args"], pars)
            packagesize = struct.pack(">i", len(payload) + 4)
            packet = packagesize + cmd.encode("ascii") + payload

            self.tcp.sendall(packet)

        except (ValueError, IndexError) as e:
            logger.warning(f"Parameter parsing error for '{cmd}' -> {pars}: {e}")
        except struct.error as e:
            logger.warning(f"Struct packing error for '{cmd}' -> {pars}: {e}")
        except BlockingIOError as e:
            # sendall() may rarely see the socket's transient non-blocking state
            # while read() is depleting the receive buffer. Retrying the whole
            # packet is avoided because sendall() may already have sent a prefix.
            # Observe, if this happens at a non-negligible scale.
            logger.warning(f"Socket temporarily not writable. Could not send '{cmd}' -> {pars}: {e}")
        except OSError as e:
            self._is_connected = False
            logger.error(f"Socket error while sending '{cmd}' -> {pars}: {e}")

    def write(self, usrstr: str):
        """
        Accepts a single user-style command string (e.g. `"vodc 1.0 5.0"`),
        splits it into command and parameters, and forwards them to `send()`.
        """
        parts = usrstr.split()
        if not parts:
            return
        self.send(*parts)


    # --- Read helpers ---

    def _recv_exact(self, count: int) -> bytes | None:
        """
        Receive `count` bytes exactly while handling TCP fragmentation. If the `recv` times out empty,
        before reading all `count` bytes, `None` is returned.
        """
        buf = bytearray()
        while len(buf) < count:
            chunk = self.tcp.recv(count - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)
    
    def _flush_rx_buffer(self, grace_timeout: float = 0.1):
        """
        Drain whatever is currently buffered, plus any tail bytes that arrive
        shortly afterwards. Best-effort recovery after packet framing loss.
        """
        old_timeout = self.tcp.gettimeout()
        try:
            self.tcp.settimeout(grace_timeout)
            while True:
                chunk = self.tcp.recv(4096)
                if not chunk:
                    break
        except (socket.timeout, BlockingIOError):
            # No more bytes arrived within grace window
            pass
        finally:
            self.tcp.settimeout(old_timeout)
    
    def _read_one_packet(self):
        """
        It's always called in the context of the `read()` function. The `read()` function
        conditionally pre-sets the TCP timeout to 0.0. Reads exactly one command-level
        reply from the RTM2.
        
        These replies always consist of:

        - 4 header bytes giving the byte-length of the reply
        - 4 command bytes to identify the reply type
        - variable length (0 to many) payload bytes
        
        The function quickly checks if the first byte of the header is present.

        - If not: it returns None
        - If yes: apply regular timeout, get the rest of the message
        
        If the function fails to retrieve the full reply after finding the first header byte,
        a `PacketFramingError` will be raised.
        """
        first_byte = self.tcp.recv(1)
        if not first_byte:
            return None

        self.tcp.settimeout(self.timeout)

        try:
            rest_header = self._recv_exact(3)
            if rest_header is None:
                return None

            header = first_byte + rest_header
            payload_size = struct.unpack(">i", header)[0] - 4

            cmd_bytes = self._recv_exact(4)
            if cmd_bytes is None:
                return None

            payload = self._recv_exact(payload_size)
            if payload is None:
                return None

            return cmd_bytes.decode("ascii"), payload

        except socket.timeout as e:
            raise PacketFramingError("Timed out while receiving packet body") from e

    def _parse_packet(self, cmd, payload):
        """
        Is called in the context of the `read()` function.
        Takes in command and payload of one reply.
        Returns a tuple of three items:
        
        - the reply `"type"`: `"state"`, `"data"`, `"raw_data"` or `None`
        - the corresponding data/values or `None`
        - an error string or `None`
        """
        cmd_def = self._COMMANDS.get(cmd)

        if not cmd_def:
            return None, None, f"Unknown incoming command: {cmd}"

        packet_type = cmd_def["type"]

        try:
            content = _decode_payload(cmd_def["reply"], payload)
        except Exception as e:
            if packet_type in {"data", "raw_data"}:
                raise PacketFramingError(
                    f"Malformed {packet_type} packet for command {cmd}"
                ) from e
            return None, None, f"Decode error for command {cmd}"

        if packet_type == "state":
            return "state", StateUpdate(cmd, content), None

        if packet_type in {"data", "raw_data"}:
            return packet_type, content, None

        return None, None, f"Unsupported packet type: {packet_type}"

    # --- The main reader ---
    
    def read(self, max_packets = 100) -> ReadResult:
        """
        Internally calls the reader helpers to formulate a response to clients.

        - Depletes the TCP Buffer
        - Parses bytes into messages
        - Parses messages into `ReadResult`
        - The return line fixes endianness and memory continuity of the numpy arrays

        `ReadResult` contains four items:

        - `updates` status update list
        - `data` numpy array
        - `raw_data` numpy array
        - `error` message

        TCP timeout management:

        - Waits regularly before capturing any message
        - After getting the first message successfully, timeout is set to 0.0
        - This is done to make the function return faster, if any data was already found
        - The timeout is reset to its intended value before returning

        `max_packets` provides a soft means to finish and not stay in an infinite loop,
        in case the RTM2 sends a lot of updates very fast.
        """
        if not self._is_connected:
            raise ConnectionError("Cannot read from device: Not connected.")

        packets = 0
        updates = []
        data = []
        raw_data = []
        error = None

        self.tcp.settimeout(self.timeout)

        try:
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
                    # Set timeout to 0.0, after the first successful packet
                    self.tcp.settimeout(0.0)

                except PacketFramingError as e:
                    frame_error = f"{e} -> Packet framing lost. Flushing receive buffer."
                    error = f"{error} Additionally: {frame_error}" if error else frame_error
                    try:
                        self._flush_rx_buffer()
                    except OSError as flush_error:
                        self._is_connected = False
                        error = f"{error} Additionally, socket error while flushing receive buffer: {flush_error}"
                    break
                except (socket.timeout, BlockingIOError):
                    # This is the default exit mode. Thrown by _read_one_packet()
                    break
                except OSError as e:
                    self._is_connected = False
                    socket_error = f"Socket error while reading: {e}"
                    error = f"{error} Additionally: {socket_error}" if error else socket_error
                    logger.error(error)
                    break

        # Reset timeout to default, in case it was set to 0.0 before
        finally:
            try:
                self.tcp.settimeout(self.timeout)
            except (AttributeError, OSError):
                pass

        return ReadResult(
            updates = updates,
            data = np.concatenate(data).astype(np.float64) if data else np.empty((0, 0)),
            raw_data = np.concatenate(raw_data).astype(np.float64) if raw_data else np.empty((0, 0)),
            error = error
        )

    def read_until(self, *terms: str, timeout: float = 10.0, listen: float = 0.0, send=None) -> ReadResult:
        """
        Repeatedly call `read()` until selected reply content appears and the
        minimum listen time has passed, an error occurs, or the outer timeout expires.

        This is a blocking convenience wrapper around the normal asynchronous RTM2
        communication model. It does not create a strict write/read transaction.

        Examples:
        ```
            rtm.read_until()
            rtm.read_until("updates")
            rtm.read_until("data", send="newd")
            rtm.read_until("updates", send="gass")
            rtm.read_until("vodc", send="vodc 0.05")
            rtm.read_until("vodc", send=("vodc", 0.05))
            rtm.read_until("vorg", listen=0.1)
        ```
        Matching terms:

            "any" / no terms  -> match any update, data, raw_data, or error
            "updates"         -> match any state update
            "data"            -> match data rows
            "raw_data"/"raw"  -> match raw data rows
            "error"           -> match an error
            other strings     -> interpreted as state-update command names, e.g. "vodc"

        `timeout` is the maximum total blocking time.

        `listen` is a minimum accumulation time counted from the start of the function,
        not from the first match. The normal successful exit condition is therefore:

            selected content has been seen AND listen seconds have passed

        If `timeout` is shorter than `listen`, `timeout` takes priority.

        All results obtained while waiting are accumulated and returned, including
        non-matching intermediate updates.

        The optional `send=` argument sends one command before waiting:

            send=None              -> do not send first
            send="newd"            -> calls self.write("newd")
            send="vodc 0.05"       -> calls self.write("vodc 0.05")
            send=("vodc", 0.05)    -> calls self.send("vodc", 0.05)
        """
        if not self._is_connected:
            raise ConnectionError("Cannot read from device: Not connected.")

        if timeout < 0:
            raise ValueError("timeout must be non-negative.")
        if listen < 0:
            raise ValueError("listen must be non-negative.")

        # Optional write-before-wait step. This keeps simple no-argument commands
        # free from Python's single-item tuple comma trap: send="gass", not send=("gass",).
        if send is not None:
            if isinstance(send, str):
                self.write(send)
            elif isinstance(send, (tuple, list)):
                if not send:
                    raise ValueError("send= tuple/list must contain at least a command name.")
                self.send(*send)
            else:
                raise TypeError("send= must be None, a command string, or a tuple/list for self.send().")

        # No terms means: match anything meaningful.
        wanted = {str(term).lower() for term in terms} if terms else {"any"}

        component_terms = {"any", "update", "updates", "data", "raw", "raw_data", "error"}
        wanted_components = wanted & component_terms
        wanted_parameters = wanted - component_terms

        def result_matches(result: ReadResult) -> bool:
            if result.error:
                # Errors always count as a match, so read_until() can return promptly.
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

        old_timeout = self.timeout

        try:
            while True:
                remaining = timeout_deadline - time.monotonic()

                if remaining <= 0:
                    break

                # The outer timeout must take priority over the socket timeout.
                if remaining < old_timeout:
                    self.timeout = remaining

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

                # Errors are exceptional transport/protocol conditions. Return promptly,
                # even if the minimum listen time has not elapsed.
                if result.error:
                    break

                if matched and time.monotonic() >= listen_deadline:
                    break

        finally:
            self.timeout = old_timeout
            try:
                self.tcp.settimeout(old_timeout)
            except (AttributeError, OSError):
                pass

        return ReadResult(
            updates=updates,
            data=np.concatenate(data) if data else np.empty((0, 0)),
            raw_data=np.concatenate(raw_data) if raw_data else np.empty((0, 0)),
            error=error,
        )
    

class RTM2Reader:
    """
    Owns one background thread that repeatedly calls `RTM2.read()`
    and forwards non-empty results into a queue.
    Intended for building fully non-blocking main applications.

    It takes an `RTM2` class instance as an argument.
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

                # Forward only meaningful results.
                if result.updates or result.data.size or result.raw_data.size or result.error:
                    self.results.put(result)

            except Exception as exc:
                # Push a synthetic error entry into the queue.
                self.results.put(exc)
                break
