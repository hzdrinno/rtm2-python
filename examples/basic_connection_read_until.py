import logging
from rtm2 import RTM2

"""
This example shows the basic long-lived connection pattern.

The RTM2 class should be instantiated when the client program starts, connected
before use, and disconnected when the client terminates.

During runtime, write() and read() / read_until() are called on-demand and
asynchronously. There is no rigid write-read correspondence. Several writes can
be clustered and replies captured with one later read.

read_until() is a blocking convenience wrapper around read(). It repeatedly
calls read() until selected reply content appears, an error occurs, or its
outer timeout expires. It still preserves the asynchronous device model and
accumulates all ReadResult contents seen while waiting.
"""

# Optional: Configure logging to see INFO messages such as connection status.
# This is useful during initial familiarization because most non-fatal command
# errors are logged rather than raised as exceptions.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Define connection parameters. HOST may be an IP address, hostname, or serial
# number if the library/device setup supports that form.
# If the IP is unknown, consider running Discover() from rtm2.
HOST = "169.254.178.185"  # or "www.MyLab.com" if port-forwarded to the RTM2
PORT = 6340
TIMEOUT = 0.2


# 1. Instantiate the device object and establish the TCP connection.
rtm = RTM2(HOST, PORT, TIMEOUT)

try:
    rtm.connect()

    # 2. Request all device settings ("gass") and wait until state updates are received.
    reply = rtm.read_until("updates", send="gass")

    if reply.error:
        print(f"Read warning/error: {reply.error}")

    print("Received setting updates:")
    for upd in reply.updates:
        print(upd)

    print("\nCurrent known device state:")
    print(rtm.get_state())

    # 3. Request new measurement data ("newd") and wait until data rows are received.
    reply = rtm.read_until("data", send="newd")

    if reply.error:
        print(f"Read warning/error: {reply.error}")

    print("\nReceived data:")
    print(reply.data)

    # 4. Several writes can be clustered. Replies and data can then be captured
    # with a single later read_until() call. The RTM2 replies in order, so use the
    # appearence of "avgt" in the ReadResult as the return trigger.
    rtm.write("viru")
    rtm.write("voru")
    rtm.write("newd")
    rtm.write("avgt 0.08")

    reply = rtm.read_until("avgt")

    if reply.error:
        print(f"Read warning/error: {reply.error}")

    print("\nReceived updates after clustered writes:")
    for upd in reply.updates:
        print(upd)

    print("\nReceived data after clustered writes:")
    print(reply.data)

finally:
    # 5. Disconnect the TCP session at the end, so other clients can connect cleanly.
    rtm.disconnect()
