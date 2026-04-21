import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import queue
import threading

# Import the RTM2 communication library
from rtm2 import RTM2, RTM2Reader, SwitState

"""
This example shows the reader thread implementation. The example uses three threads:

- the reader thread instantiated by calling `rtm2.RTM2Reader()`
- a plot in the main thread, updating as new data comes in
- a command prompt thread that allows sending further commands while the plot is updating

"""

# Optional: Configure logging to see the INFO messages (like connection status) from the library
# Very useful during initial familiarization as most non-fatal command errors don't raise exceptions.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def command_input_loop(rtm: RTM2, stop_event: threading.Event, write_lock: threading.Lock):
    """
    Read user commands from the terminal while the program is running.
    Type `'quit'` or `'exit'` to stop.
    """
    print("Running RTM2 live command line. Type `exit`/`quit` to leave.")
    while not stop_event.is_set():
        try:
            cmd = input("RTM2 command:\n").strip()
        except EOFError:
            break

        if not cmd:
            continue

        if cmd.lower() in {"quit", "exit"}:
            stop_event.set()
            break

        with write_lock:
            rtm.write(cmd)

if __name__ == '__main__':
    
    # Define connection parameters (can use S/N or IP/Hostname)
    # If the IP is unknown, consider running Discover() from rtm2
    HOST = '169.254.178.185' # or 'www.MyLab.com' if port-forwarded to the RTM2
    PORT = 6340
    TIMEOUT = 1.0 # As the reader won't stall our main thread, we can use a generous timeout

    # 1. Call the class and establish a TCP connection to the device
    rtm = RTM2(HOST, PORT, TIMEOUT)
    rtm.connect()

    # 2. Call the reader class and start the reader
    reader = RTM2Reader(rtm)
    reader.start()

    # 3. Make a thread that shows an RTM2 command prompt and that can write concurrently.
    stop_event = threading.Event()
    write_lock = threading.Lock()
    input_thread = threading.Thread(
        target=command_input_loop,
        args=(rtm, stop_event, write_lock),
        name="rtm2-input",
        daemon=True,
    )
    input_thread.start()
    
    # 4. Do some configuration 
    print("Configuring device parameters...")

    rtm.write('avgt 0.08') # Averaging time of 0.08 s
    rtm.write('vpro 0.5')   # Output voltage limit of 0.5 V
    rtm.write('ipro 0.01')  # Output current limit of 10 mA
    rtm.write('vamp 0.0')  # Drive voltage of 0.0 Vamp
    rtm.write('vodc 0.0505')  # Drive voltage of 50.5 mVDC
    
    # Define entries for the switch state list
    sw1 = SwitState([1], [2], [3], [4])
    sw2 = SwitState([5], [6], [7], [8])
    rtm.write(f'swit {sw1} {sw2}')
    
    print("Interactive example: Try changing parameters, e.g.: `vodc 0.05 10`")
    
    # 5. Run the data plotter
    fig, ax = plt.subplots()  # Create figure/axes explicitly
    plt.ion()
    plt.show(block=False)
    ax.set_xlabel("Time")
    col = 3
    ax.set_ylabel("Output Voltage")
    ax.set_title("RTM2 live data")
    last_request_time = 0.0
    

    try:
        t0 = time.time()
        while not stop_event.is_set() and plt.fignum_exists(fig.number):
            now = time.time()

            # Example automatic polling command
            if now - last_request_time > 0.2:
                with write_lock:
                    rtm.write("newd")
                last_request_time = now

            while True:
                try:
                    result = reader.results.get_nowait()
                except queue.Empty:
                    break

                if isinstance(result, Exception):
                    print(f"[RTM2] Reader thread failed: {result}")
                    stop_event.set()
                    break

                if result.error:
                    print(f"[RTM2] Read warning/error: {result.error}")

                for upd in result.updates:
                    print(f"[RTM2] {upd.parameter}: {upd.value}")

                if result.data.size:
                    ax.plot(result.data[:, 0], result.data[:, col], 'b+')
                    fig.canvas.draw()         # Redraw the figure
                    fig.canvas.flush_events() # Process GUI events so it actually shows
                
                if not result.updates and not result.data.size:
                    time.sleep(0.03)

    finally:
        stop_event.set()
        reader.stop()
        rtm.disconnect()
        plt.ioff()
        plt.show()
    
