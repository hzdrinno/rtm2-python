import time
import logging
# import numpy as np
from rtm2 import RTM2

"""
This example shows the basic implementation.
The RTM2Client class should be initialized, and a connection established when the client
program starts and should be disconnected when the client terminates

During runtime of the client, the write() and read() commands are called on-demand and asynchronously.
There is no rigid write-read correspondence. Several writes can be clustered and replies captured
with a single read. The asynchronous nature implies that a reply is sometimes not yet present when 
read() is called directly after a write() and when the TCP timeout is short. If you want to make sure
to capture the reply immediately after a write(), there are mainly two options:
1. use a larger TCP timeout, giving read() function more patience when waiting for the reply
2. wrap the read() in a small loop that stop only when actual contents are present (shown below)
"""

# Optional: Configure logging to see the INFO messages (like connection status) from the library
# Very useful during initial familiarization as most non-fatal command errors don't raise exceptions.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    
# Define connection parameters (can use S/N or IP/Hostname)
# If the IP is unknown, consider running Discover() from rtm2
HOST = '169.254.178.185' # or 'www.MyLab.com' if port-forwarded to the RTM2
PORT = 6340
TIMEOUT = 0.2

# 1. Call the class and establish a TCP connection to the device
rtm = RTM2(HOST, PORT, TIMEOUT)
rtm.connect()

# 2. rtm.read() obtains device settings and data, but only when something was requested before via rtm.write().

# Request "Get all device settings"
rtm.write('gass')
while len(rtm.read().updates) == 0: pass
print(rtm.get_state())

# Read some data.. The while loop auto-retries the read() after a timeout, if it returned empty
# The RTM2 takes longer to return the data, if there is lots of data in the device buffer
rtm.write('newd')
while not (d := rtm.read().data).size: pass
print(d)


# Use a single read() call to obtain several updates and data
rtm.write('vamp 0.01')
rtm.write('vodc 0.01')
rtm.write('avgt 0.08')
rtm.write('newd')
# But first wait a little for more new data to arrive in the buffer
time.sleep(1.0)
reply = rtm.read()
print(reply.updates)
print(reply.data)

# 3. Disconnect the TCP session at the end, so other clients can connect cleanly.
rtm.disconnect()
