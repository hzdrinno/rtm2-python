import logging
from rtm2 import RTM2

"""
This example shows compact short-script usage.

It demonstrates two API conveniences:

- the context-manager form: `with RTM2(...) as rtm:`
- the three command-writing styles:
    - structured `send()`
    - single-string `write()`
    - generated command methods under `.cmd`

The context-manager form is useful for short scripts and one-off command bursts,
because it connects at the beginning of the `with` block and disconnects when
the block exits.

For longer-running applications, see the long-lived connection example or the
threaded reader example.
"""

# Optional: Configure logging to see INFO messages such as connection status.
# This is useful during initial familiarization because most non-fatal command
# errors are logged rather than raised as exceptions.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Define connection parameters. HOST may be an IP address, hostname, or serial
# number if the library/device setup supports that form.
# If the IP is unknown, consider running Discover() from rtm2.
HOST = "169.254.178.185"  # or 'www.MyLab.com' if port-forwarded to the RTM2
PORT = 6340
TIMEOUT = 0.2


def print_reply(reply, title: str):
    """
    Print the relevant parts of a ReadResult.
    """
    if reply.error:
        print(f"Read warning/error: {reply.error}")

    if reply.updates:
        print(f"\n{title}")
        for upd in reply.updates:
            print(upd)

    if reply.data.size:
        print(f"\n{title}")
        print(reply.data)

    if reply.raw_data.size:
        print(f"\n{title}")
        print(reply.raw_data)


def main():
    # 1. Connect using the context-manager form.
    # The TCP connection is closed automatically when the block exits.
    with RTM2(HOST, PORT, TIMEOUT) as rtm:

        # 2. Request all device settings using read_until() with send=.
        reply = rtm.read_until("updates", send="gass")
        print_reply(reply, "Received setting updates:")

        print("\nCurrent known device state:")
        print(rtm.get_state())

        # 3. Structured send(): command name and arguments are passed separately.
        rtm.send("vodc", 0.02)
        reply = rtm.read_until("vodc", listen=0.1)
        print_reply(reply, "Reply after structured send():")

        # 4. Single-string write(): command and arguments are supplied as one string.
        rtm.write("vodc 0.03")
        reply = rtm.read_until("vodc", listen=0.1)
        print_reply(reply, "Reply after single-string write():")

        # 5. Generated command method: command name appears as a method under .cmd.
        rtm.cmd.vodc(0.04)
        reply = rtm.read_until("vodc", listen=0.1)
        print_reply(reply, "Reply after generated .cmd method:")

        # 6. Several writes can also be clustered, then collected with one read_until().
        rtm.send("vamp", 0.01)
        rtm.write("avgt 0.08")
        rtm.cmd.vodc(0.05)
        rtm.write("newd")

        reply = rtm.read_until("data", listen=0.1)
        print_reply(reply, "Reply after clustered writes:")


if __name__ == "__main__":
    main()