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


# Define connection parameters. HOST may be an IP address or DNS name
# The RTM2 tries to use its serial number as a DNS name by default, e.g. `RTM2-509`
# If the IP is unknown, consider running Discover() from rtm2.
HOST = "169.254.178.185"  # or 'www.MyLab.com' if port-forwarded to the RTM2
PORT = 6340
TIMEOUT = 0.2  # TCP connection and socket read timeout for this session


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

        # 7. Single send() calls can be encapsulated within read_until().
        reply = rtm.read_until("vodc", send=("vodc", 0.06), listen=0.1)
        print_reply(reply, "Reply after send() encapsulated in read_until():")


if __name__ == "__main__":
    main()