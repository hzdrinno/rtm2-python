### v1.2.3

- Fixed type stubs for the `Discover()` function.

### v1.2.2

- Added `CHANGELOG.md` and removed changes from `README.md`.
- Fixed `Discover()` quickstart example in the Readme.


### v1.2.0
- Packaged as a `src/rtm2/` Python package with generated type stubs.
- Added `pyproject.toml` packaging metadata.
- Updated TCP connection handling for IPv4/IPv6 hostname support.
- Removed `logging` from the library. 
- Tightened command error handling: malformed commands and parameters now raise exceptions.
- Reworked read-buffer draining to avoid transient non-blocking socket mode.
- TCP timeout is now set once during instantiation. Removed `rtm.timeout` public property.
- Improved `Discover()`: finds several RTMs and uses `psutil` for adapter-specific efforts.

### v1.1.0
- Added `read_until()` as a blocking convenience wrapper around `read()`. Added optional `send=` support for write-before-wait usage. Added `listen=` to continue accumulating replies for a minimum time window.
- Refactored command handling to a declarative command registry.
- Updated rampable commands to accept one or two user arguments while always encoding two doubles.
- Updated examples with clearer long-lived, context-manager, and threaded usage patterns.
- Zero-payload acknowledgements are no longer stored in the device state cache.
- Improved socket error handling and malformed payload detection.