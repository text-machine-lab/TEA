import os
import sys
import re
import time

TEA_HOME_DIR = os.path.join(*([os.path.dirname(os.path.abspath(__file__))] + [".."]*4))

sys.path.insert(0,TEA_HOME_DIR)

from subprocess import Popen, PIPE, STDOUT
from py4j.protocol import Py4JNetworkError
from py4j.java_collections import JavaArray
from py4j.java_gateway import JavaGateway, GatewayParameters
# TODO: move this out into a different directory
from code.notes.utilities.pre_processing.py4j_newsreader.gateway import GateWayServer

heideler = None

class Heideler:

    def __init__(self):

        print "\t\tHEIDELER CONSTRUCTOR: Creating Heideler Object."

        # launches java gateway server.
        GateWayServer.launch_gateway()

        print "\t\tHEIDELER CONSTRUCTOR: attempting to connect to py4j gateway"

#        self.gateway = JavaGateway(gateway_parameters=GatewayParameters(port=5007))
        self.gateway = JavaGateway()

        self.heideler = None

        init_time = time.time()

        while True:

            if time.time() - init_time > 600:
                exit("couldn't get py4j server running")

            try:
                self.heideler = self.gateway.entry_point.getHeideler()
                break

            except Py4JNetworkError:
#            except Exception as e:
#                print "Exception: ", e
                time.sleep(60)
                continue


    def process(self, document, year, month, day):

        assert month in range(1,13)

        return self.heideler.process(document, year, month, day)

if heideler is None:
    heideler = Heideler()

if __name__ == "__main__":
    h = Heideler()
    print h.process("he ran a lot yesterday", 1993, 4, 26)


