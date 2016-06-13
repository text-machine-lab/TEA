
import sys
import os

TEA_HOME_DIR = os.path.join(*([os.path.dirname(os.path.abspath(__file__))] +['..']))

sys.path.insert(0, TEA_HOME_DIR)

import time

from code.notes.utilities.pre_processing.py4j_newsreader.gateway import GateWayServer

g = GateWayServer()

g.launch_gateway()

while True:
    continue
