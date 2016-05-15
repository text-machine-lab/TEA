
import time
from code.notes.utilities.pre_processing.py4j_newsreader.gateway import GateWayServer

g = GateWayServer()

g.launch_gateway()

while True:
    time.sleep(10000)

