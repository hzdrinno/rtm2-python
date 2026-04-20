import socket
import struct
import logging
import numpy as np
from dataclasses import dataclass

"""
RTM2 Python Client Library

Requires: Python 3.9+

Users should import the RTM2 class.
The main interaction happens via non-blocking `write()` and polling `read()` calls of that class.

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
    - `read.error` is `None` or a string explaining the error
    """
    updates: list[StateUpdate]
    data: np.ndarray
    raw_data: np.ndarray
    error: None | str = None


class PacketFramingError(Exception):
    """Raised when packet framing or structural packet integrity is lost."""
    pass

# --- Command encoding/decoding helpers ---

# --- Regular single-parameter states ---
def _struct_cmd(fmt, cast_type): 
    return {
        "encode": lambda args: struct.pack(fmt, cast_type(args[0])),
        "decode": lambda payload: struct.unpack(fmt, payload)[0],
        "type": "state",
    }

# --- Rampable parameters states: 1 or 2 Double, returns 1 Double ---
def _ramp_cmd():
    def encode(args):
        if len(args) == 1:
            return struct.pack(">d", float(args[0]))
        elif len(args) == 2:
            return struct.pack(">2d", float(args[0]), float(args[1]))
        else:
            raise ValueError("Expected 1 or 2 double arguments.")

    return {
        "encode": encode,
        "decode": lambda payload: struct.unpack(">d", payload)[0],
        "type": "state",
    }

# --- parameter-less state commands ---
def _empty_cmd():
    return {
        "encode": lambda args: b'',
        "decode": lambda payload: True,  # Returns True to indicate successful state update/acknowledgment without payload
        "type": "state",
    }

# --- Data stream commands: returns parsed independently ---
def _stream_cmd():
    return {
        "encode": lambda args: b'',
        "decode": None,
        "type": "data",
    }

# --- "rawd" Raw Data streams: Sent with an integer, returns parsed independently
def _raw_stream_cmd():
    return {
        "encode": lambda args: struct.pack('>i', int(args[0])),
        "decode": None,
        "type": "raw_data",
    }

# --- "dioN" command handling: 1 Uint8 + 1 Double ---    
def _dio_cmd():
    return {
        "encode": lambda args: struct.pack(">Bd", int(args[0]), float(args[1])),
        "decode": lambda payload: struct.unpack(">Bd", payload),
        "type": "state",
    }

# --- "swit" command handling: variable length Uint32 arrays ---    
def _swit_cmd():
    def encode(args):
        pars = [int(x) for x in args]
        return struct.pack(">i", len(pars)) + struct.pack(f">{len(pars)}I", *pars)
        
    def decode(payload):
        count = struct.unpack(">i", payload[:4])[0]
        return struct.unpack(f">{count}I", payload[4:])
        
    return {"encode": encode, "decode": decode, "type": "state"}

# --- "selc" command handling: variable length Int32 arrays ---    
def _selc_cmd():
    def encode(args):
        pars = [int(x) for x in args]
        return struct.pack(">i", len(pars)) + struct.pack(f">{len(pars)}i", *pars)
        
    def decode(payload):
        count = struct.unpack(">i", payload[:4])[0]
        return struct.unpack(f">{count}i", payload[4:])
        
    return {"encode": encode, "decode": decode, "type": "state"}

# --- "puar" command handling: variable length Double arrays ---    
def _puar_cmd():
    def encode(args):
        pars = [float(x) for x in args]
        return struct.pack(">i", len(pars)) + struct.pack(f">{len(pars)}d", *pars)
        
    def decode(payload):
        count = struct.unpack(">i", payload[:4])[0]
        return struct.unpack(f">{count}d", payload[4:])
        
    return {"encode": encode, "decode": decode, "type": "state"}

# --- Other helpers meant for user interaction ---

def SwitState(SNSn: list, SNSp: list, DRVn: list, DRVp: list) -> int:
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

    for n in SNSn:
        assign_port(n, 0)
    for n in SNSp:
        assign_port(n, 8)
    for n in DRVn:
        assign_port(n, 16)
    for n in DRVp:
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
        sock.bind(('', 61557))
        
        for addr in targets:
            sock.sendto(b"UDP broadcast receive primer", (addr, primer_port))
        logger.info(f"Sent UDP broadcast receive primer. Now waiting for a UDP broadcast for {timeout} s...")

        try:
            payload, sender = sock.recvfrom(1024)
            return (payload.decode('ascii'), sender[0]) if b'RTM2' in payload else None
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

    The class also holds the TCP connection settings. The default timeout is
    1.0 seconds. If a snappier timeout is preferred, it should be supplied
    when the class is being called first (and initialized). Timeout can be
    adjusted at any time by overwriting the `.timeout` property.

    `write()` and `read()` functions are used for the device I/O. While `write()`
    is a fire-and-forget function, `read()` will always attempt to deplete the
    entire TCP buffer. It will only return after finding at least one full RTM2
    reply in the TCP buffer or after TCP timeout. 

    Generally, a continuous TCP connection is preferred. Context-manager
    support is provided and is useful for e.g. one-time configuration bursts. 
    """
    
    # --- Central command registry ---
    _COMMANDS = {
        # Doubles
        'avgt': _struct_cmd('>d', float),
        'cpro': _struct_cmd('>d', float),
        'ipro': _struct_cmd('>d', float),
        'vpro': _struct_cmd('>d', float),
        'lfrq': _struct_cmd('>d', float),
        'sres': _struct_cmd('>d', float),
        'crng': _struct_cmd('>d', float),
        'vorg': _struct_cmd('>d', float),
        'virg': _struct_cmd('>d', float),
        'phsh': _struct_cmd('>d', float),
        'time': _struct_cmd('>d', float),

        # Unsigned bytes
        'cmod': _struct_cmd('>B', int),
        'wfmd': _struct_cmd('>B', int),
        'mod?': _struct_cmd('>B', int),
        'amod': _struct_cmd('>B', int),
        'mult': _struct_cmd('>B', int),
        'refm': _struct_cmd('>B', int),
        'phlk': _struct_cmd('>B', int),
        'snsa': _struct_cmd('>B', int),
        'coax': _struct_cmd('>B', int),
        'drvp': _struct_cmd('>B', int),

        # Integers
        'meas': _struct_cmd('>i', int),

        # Parameter-less state commands
        'trig': _empty_cmd(),
        'puls': _empty_cmd(),
        'gass': _empty_cmd(),
        'srup': _empty_cmd(),
        'srdn': _empty_cmd(),
        'crup': _empty_cmd(),
        'crdn': _empty_cmd(),
        'voru': _empty_cmd(),
        'vord': _empty_cmd(),
        'viru': _empty_cmd(),
        'vird': _empty_cmd(),

        # Special structured
        'camp': _ramp_cmd(),
        'cudc': _ramp_cmd(),
        'vamp': _ramp_cmd(),
        'vodc': _ramp_cmd(),
        'dio0': _dio_cmd(),
        'dio1': _dio_cmd(),
        'swit': _swit_cmd(),
        'selc': _selc_cmd(),
        'puar': _puar_cmd(),

        # Data commands
        'newd': _stream_cmd(),
        'alld': _stream_cmd(),        
        'rawd': _raw_stream_cmd(),
    }

    def __init__(self, host: str, port: int, timeout: float = 1.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.tcp: None | socket.socket = None  # Socket is created during connect()
        self._is_connected = False
        self._state: dict[str, object] = {}

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
    
    def write(self, usrstr: str):
        """
        Splits the supplied user string into a tuple. The first component will be
        treated as the command string. Subsequent components will be treated as
        being parameters for that command.

        Malformed commands will log a warning and be ignored, but won't raise Exceptions.
        """
        if not self._is_connected:
            raise ConnectionError("Cannot write to device: Not connected.")

        try:
            cmd, *pars = usrstr.split()

            cmd_def = self._COMMANDS.get(cmd)
            if not cmd_def:
                logger.warning(f"Unknown command: {cmd}")
                return

            payload = cmd_def["encode"](pars)

            packagesize = struct.pack('>i', len(payload) + 4)
            self.tcp.sendall(packagesize + cmd.encode('ascii') + payload)

        except (ValueError, IndexError) as e:
            logger.warning(f"Parameter parsing error for '{usrstr}': {e}")
        except struct.error as e:
            logger.warning(f"Struct packing error: {e}")

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
            payload_size = struct.unpack('>i', header)[0] - 4

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
        
        - the reply type: `"state"`, `"data"`, `"raw_data"` or `None`
        - the corresponding data/values or `None`
        - an error string or `None`
        """
        cmd_def = self._COMMANDS.get(cmd)

        if not cmd_def:
            return None, None, f"Unknown incoming command: {cmd}"

        if cmd_def["type"] == "state":
            try:
                val = cmd_def["decode"](payload)
                return "state", StateUpdate(cmd, val), None
            except Exception:
                return None, None, f"Decode error for command {cmd}"

        # Process standard and raw data streams directly via NumPy mapping
        elif cmd_def["type"] in ("data", "raw_data"):
            try:
                rows, cols = struct.unpack('>ii', payload[:8])
                data = np.frombuffer(payload[8:], dtype='>d').reshape((rows, cols))
                return cmd_def["type"], data, None
            except (struct.error, ValueError) as e:
                raise PacketFramingError(
                    f"Malformed {cmd_def['type']} packet for command {cmd}"
                ) from e

        return None, None, f"Unsupported packet type: {cmd_def['type']}"

    # --- The main reader ---
    
    def read(self) -> ReadResult:
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
        """
        if not self._is_connected:
            raise ConnectionError("Cannot read from device: Not connected.")

        updates = []
        data = []
        raw_data = []
        error = None

        self.tcp.settimeout(self.timeout)

        try:
            while True:
                try:
                    packet = self._read_one_packet()
                    if packet is None:
                        break

                    packet_type, content, parse_error = self._parse_packet(*packet)
                    if error is None:
                        error = parse_error

                    if packet_type == "state":
                        self._state[content.parameter] = content.value
                        updates.append(content)
                    elif packet_type == "data":
                        if content.size:
                            data.append(content)
                    elif packet_type == "raw_data":
                        if content.size:
                            raw_data.append(content)

                    # Set timeout to 0.0, after the first successful packet
                    self.tcp.settimeout(0.0)

                except PacketFramingError as e:
                    if error is None:
                        error = f"{e} -> Packet framing lost. Flushing receive buffer."
                    self._flush_rx_buffer()
                    break
                except (socket.timeout, BlockingIOError):
                    # This is the default exit mode. Thrown by _read_one_packet()
                    break
                except OSError:
                    break

        # Reset timeout to default, in case it was set to 0.0 before
        finally:
            try:
                self.tcp.settimeout(self.timeout)
            except OSError:
                pass

        return ReadResult(
            updates = updates,
            data = np.concatenate(data).astype(np.float64) if data else np.empty((0, 0)),
            raw_data = np.concatenate(raw_data).astype(np.float64) if raw_data else np.empty((0, 0)),
            error = error
        )