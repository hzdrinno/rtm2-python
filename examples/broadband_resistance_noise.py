"""Broadband resistance noise: calibrate DRV -> SNS, then divide out drive noise.

Small AC tones establish the amplitude/phase transfer. With AC off, SNS voltage
is divided by the matched DRV current. The result retains DUT noise and both
voltage- and current-readout noise.

Before running, configure one continuous Kelvin switch state, full newd columns,
SNS/coax modes and internal-clock sine voltage drive (cmod=0, wfmd=1, phlk=0).
Choose sensible limits, fixed series resistance/ranges, a nonzero DC current well
above its fluctuations, and small calibration tones. Avoid clipping and allow
thermal settling at the bias. Edit the constants below for your DUT.

The example changes DC, AC amplitude and frequency, then restores those settings.
See docs/examples.md for the measurement principle.
"""
import time

import numpy as np
from rtm2 import RTM2


HOST = "169.254.178.185"  # IP address or DNS name, e.g. "RTM2-509".
PORT = 6340
DC_VOLTAGE_V = 1.0
CALIBRATION_AC_V = 0.05  # Peak sine amplitude, superimposed on the DC bias.
CALIBRATION_FREQUENCIES_HZ = (1250., 12500., 31250., 50000.)
CALIBRATION_ROWS = 6
POINTS_PER_CAPTURE = 500000
CAPTURES = 10
SETTLE_SECONDS = 3.0
FS = 125000.0  # rawd sample rate with decimation = 1.
EDGE_TRIM = round(0.25 * FS)


def request(rtm, command, term):
    # Unchanged settings may produce no reply.
    if (isinstance(command, tuple) and len(command) == 2 and
            rtm.get_state().get(command[0]) == command[1]):
        return
    reply = rtm.read_until(term, send=command, timeout=30.)
    if reply.error:
        raise RuntimeError(reply.error)
    if term in ("data", "raw_data") and not getattr(reply, term).size:
        raise TimeoutError(f"No {term} received for {command!r}")
    return reply


def measure_transfer(rtm, averaging_seconds):
    """Measure Z(f) = V_SNS(f) / I_DRV(f), including amplitude AND phase."""
    frequencies, impedance = [], []
    for frequency in CALIBRATION_FREQUENCIES_HZ:
        request(rtm, ("lfrq", frequency), "lfrq")
        time.sleep(SETTLE_SECONDS + 2 * averaging_seconds)
        request(rtm, "newd", "data")  # Discard buffered/transitional rows.
        time.sleep(CALIBRATION_ROWS * averaging_seconds)
        rows = request(rtm, "newd", "data").data
        if (rows.ndim != 2 or rows.shape[1] != 44 or
                not np.isfinite(rows[:, [10, 11, 24]]).all()):
            raise ValueError("Expected finite, full 44-column Kelvin data.")
        # Full native newd: frequency in column 24; complex V/I in 10 and 11.
        frequencies.append(rows[:, 24].mean())
        impedance.append((rows[:, 10] + 1j * rows[:, 11]).mean())
    return np.asarray(frequencies), np.asarray(impedance)


def equalized_current(current, resistance, frequencies, impedance):
    """Match current fluctuations to SNS with H(f) = Z(f) / R_DC."""
    mean_current = current.mean()
    # Reflection padding and subsequent edge trimming limit FFT boundary effects.
    nfft = 1 << (2 * len(current) - 1).bit_length()
    left = (nfft - len(current)) // 2
    padded = np.pad(current - mean_current,
                    (left, nfft - len(current) - left), mode="reflect")
    f = np.fft.rfftfreq(nfft, 1 / FS)
    nodes = np.r_[0., frequencies]
    response = np.r_[1., impedance / resistance]  # Unity gain at DC.
    gain = np.interp(f, nodes, np.abs(response))
    phase = np.interp(f, nodes, np.unwrap(np.angle(response)))
    h = gain * np.exp(1j * phase)
    h[-1] = h[-1].real  # The Nyquist bin must be real.
    aligned = np.fft.irfft(np.fft.rfft(padded) * h, n=nfft)
    return aligned[left:left + len(current)] + mean_current


def noise_psd(raw, frequencies, impedance):
    """Return one-sided resistance PSDs before and after drive-noise removal."""
    if (raw.ndim != 2 or raw.shape[1] < 2 or len(raw) - 2 * EDGE_TRIM < 32 or
            not np.isfinite(raw[:, :2]).all()):
        raise ValueError("Need finite voltage/current data and 32 rows after edge trimming.")
    voltage, current = raw[:, :2].T  # rawd: SNS volts, DRV amperes, output, reference.
    mean_current = current.mean()
    resistance = voltage.mean() / mean_current
    aligned_current = equalized_current(current, resistance, frequencies, impedance)

    # Central step: dividing SNS by the aligned current removes the drive noise.
    traces = np.column_stack((voltage / mean_current, voltage / aligned_current))
    traces = traces[EDGE_TRIM:-EDGE_TRIM].copy()
    traces -= traces.mean(axis=0)
    n = len(traces)
    window = np.hanning(n + 1)[:-1]  # Periodic Hann; normalize to power per Hz.
    fft = np.fft.rfft(traces * window[:, None], axis=0)
    psd = np.abs(fft)**2 / (FS * np.sum(window**2))
    psd[1:-1 if n % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(n, 1 / FS), psd.T


def main():
    with RTM2(HOST, PORT, timeout=20.) as rtm:
        request(rtm, "gass", "avgt")
        initial = rtm.get_state().copy()
        try:
            # 1. Calibrate the DRV-to-SNS response with small AC tones at the bias.
            request(rtm, ("vodc", DC_VOLTAGE_V), "vodc")
            request(rtm, ("vamp", CALIBRATION_AC_V), "vamp")
            frequencies, impedance = measure_transfer(rtm, initial["avgt"])

            # 2. Keep the DC bias, turn off AC, and acquire independent raw records.
            request(rtm, ("vamp", 0.), "vamp")
            time.sleep(SETTLE_SECONDS)
            sum_psd = None
            for _ in range(CAPTURES):
                # rawd reads buffer history: wait for a complete fresh record.
                time.sleep(POINTS_PER_CAPTURE / FS)
                raw = request(rtm, ("rawd", POINTS_PER_CAPTURE, 1), "raw_data").raw_data

                # 3. Divide out the calibrated drive contribution; average powers.
                f, psd = noise_psd(raw, frequencies, impedance)
                sum_psd = psd.copy() if sum_psd is None else sum_psd + psd
        finally:
            for name, value in (("vamp", 0.), ("vodc", initial["vodc"]),
                                ("lfrq", initial["lfrq"]), ("vamp", initial["vamp"])):
                request(rtm, (name, value), name)

    import matplotlib.pyplot as plt

    # Show only the band covered by the calibration tones. No spectral lines removed.
    band = (f >= frequencies[0]) & (f <= frequencies[-1])
    for power, label in zip(sum_psd / CAPTURES,
                            ("SNS / mean current (includes drive noise)",
                             "SNS / aligned current")):
        plt.loglog(f[band] / 1000, np.sqrt(power[band]), label=label)
    plt.xlabel("Frequency (kHz)")
    plt.xticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
    plt.tick_params(axis="x", which="minor", labelbottom=False)
    plt.ylabel("Resistance ASD (Ohm/sqrt(Hz))")
    plt.grid(which="both", alpha=.2)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
