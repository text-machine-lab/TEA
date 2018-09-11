"""Used for interacting with newsreader components.
"""
import sys
import os
import subprocess
import time
import signal
import atexit

TEA_HOME_DIR = os.path.join(*([os.path.dirname(os.path.abspath(__file__))]+[".."]*5))
from src.config import env_paths

if env_paths()["PY4J_DIR_PATH"] is None:
    sys.exit("PY4J_DIR_PATH environment variable not specified")

from py4j.java_gateway import GatewayClient

PY4J_DEPENDENCIES="{}/*".format(env_paths()["PY4J_DIR_PATH"])
TOK_JAR_PATH=TEA_HOME_DIR + "/dependencies/NewsReader/ixa-pipes-1.1.0/ixa-pipe-tok-1.8.2.jar"
POS_JAR_PATH=TEA_HOME_DIR + "/dependencies/NewsReader/ixa-pipes-1.1.0/ixa-pipe-pos-1.4.1.jar"
NER_JAR_PATH=TEA_HOME_DIR + "/dependencies/NewsReader/ixa-pipes-1.1.0/ixa-pipe-nerc-1.5.2.jar"
PARSE_JAR_PATH=TEA_HOME_DIR + "/dependencies/NewsReader/ixa-pipes-1.1.0/ixa-pipe-parse-1.1.0.jar"
SRC_DR=TEA_HOME_DIR + "/src/notes/utilities/pre_processing/py4j_newsreader"
DEPENDENCIES=":{}:{}:{}:{}:{}:{}:{}/*".format(PY4J_DEPENDENCIES, TOK_JAR_PATH, POS_JAR_PATH, NER_JAR_PATH, PARSE_JAR_PATH, SRC_DR, SRC_DR)
#print("dependencies", DEPENDENCIES)

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
            print("launching gateway")
            GateWayServer.server = subprocess.Popen(["java", "-cp", DEPENDENCIES, "gateway.GateWay"], encoding='utf8')

        else:
            print("py4j server is already running")

    @staticmethod
    @atexit.register
    def cleanup():

        print("terminating py4j server")
        if GateWayServer.server is not None:

            os.kill(GateWayServer.server.pid, signal.SIGKILL)
            GateWayServer.server = None

    def __del__(self):
        pass

if __name__ == "__main__":
    pass

