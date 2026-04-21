import time
import logging
# import numpy as np
import matplotlib.pyplot as plt

# Import the RTM2 communication library
from rtm2 import RTM2, SwitState

"""
This example shows the context-manager implementation.
The `with` block first applies some config settings, flushes old data and then reads 
data as the measurement happens. It handles connection and disconnection implicitly.

To be used with: Short encapsulated programs, where no connection to be RTM2 must
be maintained between measurements.
"""

# Optional: Configure logging to see the INFO messages (like connection status) from the library
# Very useful during initial familiarization as most non-fatal command errors don't raise exceptions.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def data_plotter(device, period=2.0, runtime=60, col=17):
    """
    A basic example data plotter
    Sends regular 'newd' writes and reads to the device to stream & plot new data.
    """
    print(f"Starting data plotter for {runtime} seconds... Close the plot window or press Ctrl+C to stop.")
    
    fig, ax = plt.subplots()  # Create figure/axes explicitly
    plt.ion()
    plt.show(block=False)

    try:
        t0 = time.time()

        while time.time() - t0 < runtime:
            device.write('newd')
            plt.pause(period)
            newdata = device.read().data

            if newdata.size:
                ax.plot(newdata[:, 0], newdata[:, col], color='blue')
                fig.canvas.draw()         # Redraw the figure
                fig.canvas.flush_events() # Process GUI events so it actually shows

    except KeyboardInterrupt:
        print("\nPlotting manually aborted by user.")
    finally:
        plt.ioff()
        plt.show(block=False)  # Blocks here intentionally — close the window to continue

if __name__ == '__main__':
    
    # Define connection parameters (can use S/N or IP/Hostname)
    HOST = '169.254.178.185' # or 'www.MyLab.com' if port-forwarded to the RTM2
    PORT = 6340
    TIMEOUT = 0.2
    
    # 1. Connect using the Context Manager ('with' statement)
    # This guarantees the socket is closed gracefully when the script finishes or crashes.
    with RTM2(HOST, PORT, TIMEOUT) as rtm:
        
        print("Configuring device parameters...")
        
        # 2. Setup some functional parameters
        rtm.write('avgt 0.08') # Averaging time of 0.08 s
        rtm.write('vpro 0.5')   # Output voltage limit of 0.5 V
        rtm.write('ipro 0.01')  # Output current limit of 10 mA
        rtm.write('vamp 0.0')  # Drive voltage of 0.0 Vamp
        rtm.write('vodc 0.05')  # Drive voltage of 0.05 VDC
        
        # Define entries for the switch state list
        sw1 = SwitState([1], [2], [1], [2])
        sw2 = SwitState([5], [6], [5], [6])
        rtm.write(f'swit {sw1} {sw2}')
       
        # 3. We also request 'newd' once, to flush all old data
        # We call .read() to clear them out, updating the rtm properties in the process.
        rtm.write('newd')
        u = []
        d = []
        while len(d) == 0:
            reply = rtm.read()
            u = reply.updates
            d = reply.data
        print(u)    
        print(d)
        
        # 4. Run the data plotter
        # Initialize the demo plotter to run for 20 s, updating every 1 s. 
        # It will plot data column 3 which is "Voltage Output DC"
        data_plotter(rtm, period=1.0, runtime=20, col=3)
        
    print("Execution complete. Context Manager has safely disconnected the device.")