# RTM2 Python command reference

[Back to the README](../README.md)

RTM2 commands are asynchronous: `send()`, `write()`, and `cmd.*()` transmit a command and return `None`. Retrieve replies with `read()` or `read_until()`.

The three command interfaces encode the same protocol packet:

```python
rtm.cmd.vodc(1.0, 10.0)
rtm.send("vodc", 1.0, 10.0)
rtm.write("vodc 1.0 10.0")
```

Integer command arguments are encoded as signed 32-bit values unless marked `uint8` or `uint32`. Floating-point arguments are encoded as 64-bit doubles. Multi-value commands send a signed 32-bit count followed by their values.

When the optional arrival time of a rampable command is omitted, the library sends `0.0`. When `rawd` decimation is omitted, the library sends `1`.

## Commands

| Command | Python arguments | Reply | Purpose |
| --- | --- | --- | --- |
| `avgt` | `value: float` | `float` | Set averaging time. |
| `cpro` | `value: float` | `float` | Set current limit. |
| `ipro` | `value: float` | `float` | Set current limit. |
| `vpro` | `value: float` | `float` | Set output-voltage limit. |
| `lfrq` | `value: float` | `float` | Set AC frequency. |
| `sres` | `value: float` | `float` | Set series resistance. Negative values enable automatic selection. |
| `crng` | `value: float` | `float` | Set current-measurement range. Negative values enable automatic selection. |
| `vorg` | `value: float` | `float` | Set output-voltage measurement range. Negative values enable automatic selection. |
| `virg` | `value: float` | `float` | Set input-voltage measurement range. Negative values enable automatic selection. |
| `phsh` | `value: float` | `float` | Set AC phase shift from the reference input. |
| `time` | `timestamp: float` | `float` | Request the difference between the supplied timestamp and the RTM2 internal timestamp. |
| `cmod` | `value: int` (`uint8`) | `int` (`uint8`) | Set output-control mode. |
| `wfmd` | `value: int` (`uint8`) | `int` (`uint8`) | Set waveform mode. |
| `modq` | `value: int` (`uint8`) | `int` (`uint8`) | Detected analysis mode. |
| `amod` | `value: int` (`uint8`) | `int` (`uint8`) | Set analysis mode. |
| `mult` | `value: int` (`uint8`) | `int` (`uint8`) | Set multisample mode. |
| `refm` | `value: int` (`uint8`) | `int` (`uint8`) | Set reference-multiplexer input. |
| `phlk` | `value: int` (`uint8`) | `int` (`uint8`) | Set phase-locking behavior. |
| `snsa` | `value: int` (`uint8`) | `int` (`uint8`) | Set SNS preamplifier mode. |
| `coax` | `value: int` (`uint8`) | `int` (`uint8`) | Set BNC coax mode. |
| `drvp` | `value: int` (`uint8`) | `int` (`uint8`) | Set drive-polarity mode. |
| `meas` | `count: int` | `int` | Set the data-sample counter. Negative values enable infinite sampling. |
| `trig` | none | acknowledgement | Begin a new demodulation window immediately. |
| `puls` | none | acknowledgement | Begin pulse-train or arbitrary-waveform generation. |
| `gass` | none | acknowledgement | Request all device settings. |
| `cldt` | none | acknowledgement | Clear the device-side data buffer. |
| `srup` | none | acknowledgement | Increase series resistance. |
| `srdn` | none | acknowledgement | Decrease series resistance. |
| `crup` | none | acknowledgement | Increase current-measurement range. |
| `crdn` | none | acknowledgement | Decrease current-measurement range. |
| `voru` | none | acknowledgement | Increase output-voltage measurement range. |
| `vord` | none | acknowledgement | Decrease output-voltage measurement range. |
| `viru` | none | acknowledgement | Increase input-voltage measurement range. |
| `vird` | none | acknowledgement | Decrease input-voltage measurement range. |
| `camp` | `value: float`, optional `arrival_time: float = 0.0` | `float` | Set current-amplitude setpoint, optionally ramped over the arrival time. |
| `cudc` | `value: float`, optional `arrival_time: float = 0.0` | `float` | Set current-DC setpoint, optionally ramped over the arrival time. |
| `vamp` | `value: float`, optional `arrival_time: float = 0.0` | `float` | Set voltage-amplitude setpoint, optionally ramped over the arrival time. |
| `vodc` | `value: float`, optional `arrival_time: float = 0.0` | `float` | Set voltage-DC setpoint, optionally ramped over the arrival time. |
| `dio0` | `mode: int` (`uint8`), `value: float` | `tuple[int, float]` | Set DIO0 mode. |
| `dio1` | `mode: int` (`uint8`), `value: float` | `tuple[int, float]` | Set DIO1 mode. |
| `swit` | `*states: int` (`uint32`) | `tuple[int, ...]` | Define switch-matrix states. |
| `selc` | `*indices: int` | `tuple[int, ...]` | Select data-channel indices returned by `newd`. |
| `puar` | `*values: float` | `tuple[float, ...]` | Set pulse-parameter array entries. |
| `newd` | none | data matrix | Request all previously unsent data rows. |
| `alld` | none | data matrix | Request all data rows. |
| `rawd` | `rows: int`, optional `decimation: int = 1` | raw-data matrix | Request raw ADC rows. The reply contains every `decimation`-th sample from the device-side data buffer. |

Data replies are exposed as NumPy matrices through `ReadResult.data` and `ReadResult.raw_data`. State replies are available as `StateUpdate` entries and through `RTM2.get_state()` after they have been read.
