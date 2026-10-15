"""Numerical and reply-validation checks for the broadband noise example."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location(
    "noise_example", ROOT / "examples" / "broadband_resistance_noise.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class ReplyValidationTests(unittest.TestCase):
    def test_unchanged_setpoints_return_without_sending_or_waiting(self):
        device = Mock()
        device.get_state.return_value = dict(vamp=0., vodc=1., lfrq=1250.)
        for name, value in device.get_state.return_value.items():
            with self.subTest(name=name):
                self.assertIsNone(example.request(device, (name, value), name))
        device.read_until.assert_not_called()

    def test_changed_setpoints_and_measurement_requests_are_sent(self):
        device = Mock()
        device.get_state.return_value = dict(vamp=0.)
        reply = SimpleNamespace(error=None, data=np.ones((1, 44)),
                                raw_data=np.ones((100, 4)))
        device.read_until.return_value = reply
        for command, term in ((("vamp", .05), "vamp"), ("newd", "data"),
                              (("rawd", 500000, 1), "raw_data")):
            with self.subTest(command=command):
                device.read_until.reset_mock()
                self.assertIs(example.request(device, command, term), reply)
                device.read_until.assert_called_once_with(term, send=command, timeout=30.)

    def test_empty_measurement_reply_raises_even_without_reply_error(self):
        device = Mock()
        device.read_until.return_value = SimpleNamespace(
            error=None, data=np.empty((0, 0)), raw_data=np.empty((0, 0)))
        for command, term in (("newd", "data"), (("rawd", 500000, 1), "raw_data")):
            with self.subTest(command=command):
                with self.assertRaisesRegex(TimeoutError, f"No {term} received"):
                    example.request(device, command, term)

    def test_protocol_errors_are_reported(self):
        device = Mock()
        device.read_until.return_value = SimpleNamespace(error="Malformed data")
        with self.assertRaisesRegex(RuntimeError, "Malformed data"):
            example.request(device, "newd", "data")

    def test_unusable_kelvin_data_is_rejected(self):
        nonfinite = np.ones((2, 44))
        nonfinite[0, 10] = np.nan
        for rows in (np.ones((2, 20)), nonfinite):
            with self.subTest(shape=rows.shape):
                reply = SimpleNamespace(data=rows)
                with patch.object(example, "request", return_value=reply), \
                     patch.object(example.time, "sleep"):
                    with self.assertRaisesRegex(ValueError, "Kelvin data"):
                        example.measure_transfer(Mock(), .16)

    def test_unusable_or_too_short_raw_records_are_rejected(self):
        short = np.ones((2 * example.EDGE_TRIM + 31, 4))
        nonfinite = np.ones((2 * example.EDGE_TRIM + 32, 4))
        nonfinite[0, 0] = np.nan
        for raw in (np.empty((0, 0)), short, nonfinite):
            with self.subTest(shape=raw.shape):
                with self.assertRaisesRegex(ValueError, "32 rows after edge trimming"):
                    example.noise_psd(raw, np.array([1250., 50000.]),
                                      np.array([330., 330.]))

class NoiseNumericalTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(125000) / example.FS
        self.frequencies = np.array([1000., 50000.])
        self.flat_impedance = np.full(2, 330. + 0j)

    def integrated_power(self, raw, impedance=None):
        if impedance is None:
            impedance = self.flat_impedance
        f, psd = example.noise_psd(raw, self.frequencies, impedance)
        return psd.sum(axis=1) * (f[1] - f[0])

    def test_equalization_matches_amplitude_and_phase_preserving_dc(self):
        tau = .8e-6
        frequency = 20000.
        current = 1e-4 + 1e-5 * np.cos(2 * np.pi * frequency * self.t)
        gain = 1 + .2 * self.frequencies / 50000.
        impedance = 330 * gain * np.exp(-2j * np.pi * self.frequencies * tau)
        actual = example.equalized_current(current, 330., self.frequencies, impedance)
        expected = (1e-4 + 1e-5 * (1 + .2 * frequency / 50000.) *
                    np.cos(2 * np.pi * frequency * (self.t - tau)))
        interior = slice(example.EDGE_TRIM, -example.EDGE_TRIM)
        np.testing.assert_allclose(actual[interior], expected[interior], atol=2e-10, rtol=0)
        self.assertAlmostEqual(actual[interior].mean(), 1e-4, places=10)

    def test_transfer_correction_removes_drive_noise_with_gain_and_delay(self):
        tau = .8e-6
        current = 1e-4 * (1 + .05 * np.cos(2 * np.pi * 20000 * self.t))
        voltage = .033 * (1 + .05 * 1.08 * np.cos(2 * np.pi * 20000 * (self.t - tau)))
        gain = 1 + .2 * self.frequencies / 50000.
        impedance = 330 * gain * np.exp(-2j * np.pi * self.frequencies * tau)
        before, after = self.integrated_power(np.column_stack((voltage, current)), impedance)
        # Direct V/I leaves drive noise when the two channels have different phase.
        unaligned = voltage / current
        self.assertGreater(unaligned.std(), .1)
        self.assertLess(after / before, 1e-10)

    def test_division_recovers_dut_fluctuations_under_drive_modulation(self):
        current = 1e-4 * (1 + .05 * np.sin(2 * np.pi * 1000 * self.t))
        delta_r = .001 * np.cos(2 * np.pi * 3000 * self.t)
        raw = np.column_stack((current * (330 + delta_r), current))
        before, after = self.integrated_power(raw)
        self.assertGreater(before / after, 1e8)
        self.assertAlmostEqual(after / (.001**2 / 2), 1., places=5)

    def test_dut_noise_survives_anticorrelated_voltage_and_current(self):
        resistance = 330 + .01 * np.cos(2 * np.pi * 3000 * self.t)
        current = 1. / (100 + resistance)
        voltage = resistance * current
        self.assertLess(np.corrcoef(voltage, current)[0, 1], -.999)
        _, after = self.integrated_power(np.column_stack((voltage, current)))
        self.assertAlmostEqual(after / (.01**2 / 2), 1., places=5)

    def test_voltage_and_current_readout_noise_remain(self):
        tone = np.cos(2 * np.pi * 3000 * self.t)
        for voltage_noise, current_noise in ((1e-7, 0.), (0., 1e-9)):
            with self.subTest(voltage_noise=voltage_noise, current_noise=current_noise):
                raw = np.column_stack((.033 + voltage_noise * tone,
                                       1e-4 + current_noise * tone))
                _, after = self.integrated_power(raw)
                expected = ((voltage_noise - 330 * current_noise) / 1e-4)**2 / 2
                self.assertAlmostEqual(after / expected, 1., places=5)

    def test_one_sided_psd_normalization_for_even_and_odd_lengths(self):
        for count in (125000, 125001):
            with self.subTest(count=count):
                amplitude = .001
                current = np.full(count, 1e-4)
                voltage = current * (330 + amplitude * (-1.)**np.arange(count))
                power = self.integrated_power(np.column_stack((voltage, current)))
                np.testing.assert_allclose(power / amplitude**2, 1., rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
