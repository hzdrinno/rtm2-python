import time
import logging
from rtm2 import RTM2

"""
This example shows the different ways to write commands.
    - structured `send()`
    - single-string `write()`
    - dedicated method `.cmd.xxxx()`
"""

# Optional: Configure logging to see the INFO messages (like connection status) from the library
# Very useful during initial familiarization as most non-fatal command errors don't raise exceptions.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    
# Define connection parameters (can use S/N or IP/Hostname)
# If the IP is unknown, consider running Discover() from rtm2
HOST = '169.254.178.185' # or 'www.MyLab.com' if port-forwarded to the RTM2
PORT = 6340
TIMEOUT = 0.2

# Call the class and establish a TCP connection to the device
rtm = RTM2(HOST, PORT, TIMEOUT)
rtm.connect()

# rtm.read() obtains device settings and data, but only when something was requested before via e.g. rtm.write().

# Request "Get all device settings"
rtm.write('gass')
time.sleep(0.5)
while not rtm.read().updates: pass
print(rtm.get_state())

# Do three successive fire-and-forget writes:
# 1. Structured `send()` approach
rtm.send('vodc', 0.02)
time.sleep(0.5)

# 2. The single-string `write()` approach
rtm.write('vodc 0.03')
time.sleep(0.5)

# 3. The dedicated method `.cmd.vodc()` approach
rtm.cmd.vodc(0.04)
time.sleep(0.5)

# Read the three successive state changes.
for upd in rtm.read().updates: print(upd)

# Disconnect
rtm.disconnect()
