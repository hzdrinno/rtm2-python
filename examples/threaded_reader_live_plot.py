import time
import logging
import queue
import threading
import matplotlib.pyplot as plt

from rtm2 import RTM2, RTM2Reader, SwitState

"""
This example shows a threaded application-style client.

The main program uses three execution paths:

- the RTM2Reader thread, which repeatedly calls rtm.read()
- the main thread, which requests data and updates the plots
- a command-input thread, which allows live user commands while plotting

When RTM2Reader is used, user code should not call rtm.read() or
rtm.read_until() directly anymore. All received replies should be consumed
from reader.results.
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
TIMEOUT = 1.0  # The reader thread can use a generous timeout without blocking the plot.


def command_input_loop(rtm: RTM2, stop_event: threading.Event, write_lock: threading.Lock):
    """
    Read user commands from the terminal while the program is running.
    Type 'quit' or 'exit' to stop.
    """
    print("Running RTM2 live command line. Type 'exit' or 'quit' to leave.")

    while not stop_event.is_set():
        try:
            cmd = input("RTM2 command:\n").strip()
        except EOFError:
            stop_event.set()
            break

        if not cmd:
            continue

        if cmd.lower() in {"quit", "exit"}:
            stop_event.set()
            break

        with write_lock:
            rtm.write(cmd)


def main():
    # 1. Instantiate the device object and helper objects.
    rtm = RTM2(HOST, PORT, TIMEOUT)
    reader = RTM2Reader(rtm)
    stop_event = threading.Event()
    write_lock = threading.Lock()
    input_thread = None
    fig = None

    try:
        # 2. Establish the TCP connection, deplete previous data and start the reader thread.
        rtm.connect()
        rtm.read_until("data", send="newd")
        reader.start()

        # 3. Start a small command prompt thread for interactive writes.
        # The write_lock prevents two user/application writes from overlapping.
        input_thread = threading.Thread(
            target=command_input_loop,
            args=(rtm, stop_event, write_lock),
            name="rtm2-input",
            daemon=True,
        )
        input_thread.start()

        # 4. Send initial configuration commands.
        # These are fire-and-forget writes. Their confirmations are received by
        # the RTM2Reader thread and later consumed from reader.results.
        print("Configuring device parameters...")

        with write_lock:
            rtm.write("avgt 0.08")    # Averaging time of 0.08 s
            rtm.write("vpro 0.5")     # Output voltage limit of 0.5 V
            rtm.write("ipro 0.01")    # Output current limit of 10 mA
            rtm.write("cmod 0")       # Control mode 0: Setpoints map directly to DRV Voltage
            rtm.write("vamp 0.0")     # Drive voltage amplitude of 0.0 V
            rtm.write("vodc 0.0505")  # Drive voltage DC setpoint of 50.5 mV
            rtm.write("sres 100")     # Series Resistor of 100 Ohm

            sw1 = SwitState([1], [2], [3], [4])
            sw2 = SwitState([5], [6], [7], [8])
            rtm.write(f"swit {sw1} {sw2}")

        print("Interactive example: try changing parameters, e.g. 'vodc 0.05 10'.")

        # 5. Set up one vertically oriented figure containing the live data plot
        # and a text panel for the current device state.
        plt.ion()

        fig, (ax, state_ax) = plt.subplots(
            2,
            1,
            figsize=(8, 9),
            gridspec_kw={"height_ratios": [3, 1]},
        )
        fig.subplots_adjust(hspace=0.45)
        fig.canvas.manager.set_window_title("RTM2 live data and state")
        plt.show(block=False)

        col = 3
        ax.set_xlabel("Time")
        ax.set_ylabel("Output Voltage")
        ax.set_title("RTM2 live data")
        ax.grid(True, which="major", color="lightgrey")

        state_ax.set_title("Current RTM2 state")
        state_ax.axis("off")
        state_text = state_ax.text(
            0.02,
            0.98,
            "Waiting for state updates...",
            va="top",
            ha="left",
            family="monospace",
        )

        live_state = {}
        last_request_time = 0.0

        # 6. Main application loop.
        # The main thread periodically requests new data, quickly drains the
        # reader queue, then processes the collected results and redraws once.
        while not stop_event.is_set() and plt.fignum_exists(fig.number):
            now = time.monotonic()

            if now - last_request_time > 0.2:
                with write_lock:
                    rtm.write("newd")
                last_request_time = now

            queued_results = []
            state_changed = False
            data_changed = False

            while True:
                try:
                    queued_results.append(reader.results.get_nowait())
                except queue.Empty:
                    break

            for result in queued_results:
                if isinstance(result, Exception):
                    print(f"[RTM2] Reader thread failed: {result}")
                    stop_event.set()
                    break

                if result.error:
                    print(f"[RTM2] Read warning/error: {result.error}")

                if result.updates:
                    for upd in result.updates:
                        live_state[upd.parameter] = upd.value
                    state_changed = True

                if result.data.size:
                    ax.plot(result.data[:, 0], result.data[:, col], 'b+')
                    data_changed = True

            if state_changed:
                state_text.set_text(
                    "\n".join(f"{key}: {value}" for key, value in sorted(live_state.items()))
                )

            if state_changed or data_changed:
                fig.canvas.draw()
                fig.canvas.flush_events()

            plt.pause(0.03)

    finally:
        # 7. Stop helper threads and disconnect cleanly.
        stop_event.set()
        reader.stop()
        rtm.disconnect()

        if fig is not None:
            plt.ioff()
            plt.show()


if __name__ == "__main__":
    main()
