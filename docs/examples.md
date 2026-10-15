# Examples

Run these scripts from a repository checkout after installing the package:

```bash
python -m pip install -e ".[examples]"
```

Edit the connection settings at the top of the chosen script. Matplotlib is
needed for the plotting examples.

| Example | What it teaches |
| --- | --- |
| [Basic connection](../examples/basic_connection_read_until.py) | A long-lived connection with explicit connect/disconnect and `read_until()` |
| [Context manager and commands](../examples/context_manager_commands.py) | Short scripts with `with RTM2(...)` and the three command-writing styles |
| [Threaded reader and live plot](../examples/threaded_reader_live_plot.py) | A background reader with a responsive plotting and command-input loop |
| [Broadband resistance noise](../examples/broadband_resistance_noise.py) | Measure the DRV-to-SNS transfer first, then divide drive noise out of a DC-biased SNS capture |

## Basic connection

`basic_connection_read_until.py` shows the connection lifecycle, requests settings
and measurement data, and groups several writes before collecting replies.
`read_until()` waits for selected reply content while retaining all replies read
along the way.

## Context manager and commands

`context_manager_commands.py` uses `with RTM2(...) as rtm` to connect and disconnect
automatically. It demonstrates structured `send()`, string-based `write()`, and
generated `.cmd.*()` methods, including a command sent directly by `read_until()`.

## Threaded reader and live plot

`threaded_reader_live_plot.py` lets `RTM2Reader` own the socket reads while the main
loop plots data, displays state and accepts commands. Consume the reader's result
queue; keep `read()` and `read_until()` out of the main loop while it is active.

## Broadband resistance noise

RTM2 normally measures slow resistance fluctuations directly over time using AC
excitation, including in the mHz range. This example demonstrates a complementary
measurement: a classical broadband noise spectrum in the kHz range under DC bias.

The key is to **measure the complex DRV-to-SNS transfer function first**. Drive
noise appears in both the DUT current and the sensed voltage. Matching their
amplitude and phase response before division removes that common contribution.
The corrected spectrum contains DUT resistance noise and the noise of both the
voltage and current measurements.

### Measurement sequence

1. Apply small AC tones on the chosen DC bias. The native Kelvin `newd` results
   give the DRV-to-SNS response as the complex impedance
   `Z(f) = V_SNS(f) / I_DRV(f)`.
2. Turn off the AC excitation, keeping the DC bias and measurement configuration.
   Capture simultaneous SNS voltage and DRV current with `rawd` at 125 kS/s.
3. Match the current fluctuations to the SNS response with `H(f) = Z(f) / R_DC`,
   where `R_DC = mean(V_SNS) / mean(I_DRV)`. Keep the DC current unchanged, then
   form `R(t) = V_SNS(t) / I_aligned(t)` and calculate its noise spectrum.

For small fluctuations, the division gives
`delta_R(t) ~= [delta_V_SNS(t) - R_DC * delta_I_aligned(t)] / I_DC`.
The drive-induced terms cancel because the independently measured transfer
matches their response. Simply dividing the two unaligned raw traces can leave
drive noise when their gain or phase differs. The transfer is measured with tones,
so the correction does not learn a fit from the noise it is meant to measure.

### Running the example

Read the setup assumptions in the script header and edit its constants for the
DUT, particularly the bias and calibration amplitude. Allow thermal settling at
that bias, and keep the topology, series resistance and measurement ranges fixed
between calibration and noise acquisition.

```bash
python examples/broadband_resistance_noise.py
```

The plot compares SNS voltage divided by the mean current (still containing drive
noise) with SNS divided by the aligned current. Both curves are in
Ohm/sqrt(Hz). Each record is demeaned and Hann-windowed; spectra are averaged in
power before taking the square root. The script waits for fresh buffer history
between captures and trims filter edges. It retains spectral lines.

The example assumes a linear, predominantly resistive DUT and a smooth response
between calibration tones. It interpolates transfer magnitude and unwrapped phase,
anchors the normalized response at unity at DC, and plots only the band spanned by
the tones (1.25–50 kHz by default). Add tones to resolve a more complicated response,
and recalibrate after changing the bias, DUT, cabling, topology or ranges.

Drive-noise cancellation depends on an accurate, stable transfer calibration.
Voltage- and current-readout noise remain in the corrected result, alongside DUT
noise; calibration error can leave residual drive noise.
