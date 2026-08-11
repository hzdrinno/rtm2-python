import struct
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtm2 import RTM2, _decode_payload  # noqa: E402


class RecordingSocket:
    def __init__(self):
        self.packets = []

    def sendall(self, packet):
        self.packets.append(packet)


def command_packet(command, payload=b""):
    return struct.pack(">i", len(payload) + 4) + command.encode("ascii") + payload


class CommandEncodingTests(unittest.TestCase):
    def setUp(self):
        self.rtm = RTM2("example.invalid", 6340)
        self.socket = RecordingSocket()
        self.rtm.tcp = self.socket
        self.rtm._is_connected = True

    def assert_last_packet(self, command, payload=b""):
        self.assertEqual(self.socket.packets[-1], command_packet(command, payload))

    def test_rawd_default_is_encoded_by_all_command_interfaces(self):
        calls = (
            lambda: self.rtm.cmd.rawd(25),
            lambda: self.rtm.send("rawd", 25),
            lambda: self.rtm.write("rawd 25"),
        )

        for call in calls:
            with self.subTest(call=call):
                self.socket.packets.clear()
                call()
                self.assert_last_packet("rawd", struct.pack(">ii", 25, 1))

    def test_rawd_explicit_decimation_is_preserved(self):
        self.rtm.send("rawd", 25, 4)
        self.assert_last_packet("rawd", struct.pack(">ii", 25, 4))

    def test_rawd_rejects_invalid_argument_counts(self):
        with self.assertRaises(ValueError):
            self.rtm.send("rawd")

        with self.assertRaises(ValueError):
            self.rtm.send("rawd", 25, 2, 1)

    def test_rawd_rejects_values_outside_int32(self):
        with self.assertRaises(ValueError):
            self.rtm.send("rawd", 2**31)

    def test_rampable_command_still_defaults_arrival_time_to_zero(self):
        self.rtm.cmd.camp(1.5)
        self.assert_last_packet("camp", struct.pack(">dd", 1.5, 0.0))

    def test_parameterless_command_packet(self):
        self.rtm.cmd.trig()
        self.assert_last_packet("trig")

    def test_counted_sequence_packet(self):
        self.rtm.cmd.selc(1, 3, 5)
        payload = struct.pack(">i", 3) + struct.pack(">iii", 1, 3, 5)
        self.assert_last_packet("selc", payload)


class ReplyDecodingTests(unittest.TestCase):
    def test_data_matrix_is_decoded(self):
        payload = struct.pack(">ii", 2, 2) + struct.pack(">dddd", 1.0, 2.0, 3.0, 4.0)

        matrix = _decode_payload("data", payload)

        np.testing.assert_array_equal(matrix, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_malformed_data_matrix_is_rejected(self):
        payload = struct.pack(">ii", 2, 2) + struct.pack(">d", 1.0)

        with self.assertRaises(ValueError):
            _decode_payload("data", payload)


if __name__ == "__main__":
    unittest.main()
