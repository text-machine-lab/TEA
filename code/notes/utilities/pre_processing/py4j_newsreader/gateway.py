"""Used for interacting with newsreader components.
"""

import os
import subprocess
import time
import signal
import atexit

from py4j.java_gateway import GatewayClient

if "PY4J_DIR_PATH" not in os.environ:

    exit("please defined PY4J_DIR_PATH")

if "TEA_PATH" not in os.environ:

    exit("please define TEA_PATH")

PY4J_DEPENDENCIES="{}/*".format(os.environ["PY4J_DIR_PATH"])
TOK_JAR_PATH=os.environ["TEA_PATH"] + "/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-tok-1.8.2.jar"
POS_JAR_PATH=os.environ["TEA_PATH"] + "/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-pos-1.4.1.jar"
NER_JAR_PATH=os.environ["TEA_PATH"] + "/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-nerc-1.5.2.jar"
PARSE_JAR_PATH=os.environ["TEA_PATH"] + "/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-parse-1.1.0.jar"
SRC_DR=os.environ["TEA_PATH"] + "/code/notes/utilities/pre_processing/py4j_newsreader"
DEPENDENCIES=":{}:{}:{}:{}:{}:{}:{}/*".format(PY4J_DEPENDENCIES, TOK_JAR_PATH, POS_JAR_PATH, NER_JAR_PATH, PARSE_JAR_PATH, SRC_DR, SRC_DR)

class GateWayServer(object):
    """creates the py4j gateway to allow access to jvm objects.
    only one gateway server may be running at a time on a specific port.
    """

    server = None

    def __init__(self):
        pass

    @staticmethod
    def launch_gateway():

        if GateWayServer.server is None:
            print "launching gateway"
            GateWayServer.server = subprocess.Popen(["java", "-cp", DEPENDENCIES, "gateway.GateWay"])

        else:
            print "py4j server is already running"

    @staticmethod
    @atexit.register
    def cleanup():

        print "terminating py4j server"
        if GateWayServer.server is not None:

            os.kill(GateWayServer.server.pid, signal.SIGKILL)
            GateWayServer.server = None

    def __del__(self):
        pass

if __name__ == "__main__":

    print "nothing to do in main"
    GateWayServer.launch_gateway()

    while True:
        pass

