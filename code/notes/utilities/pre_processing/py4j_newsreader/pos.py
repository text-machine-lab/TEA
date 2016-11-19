"""Allows newsreader pos component to be accessed by python.
"""

from subprocess import Popen, PIPE, STDOUT

from py4j.java_collections import JavaArray
from py4j.java_gateway import JavaGateway

import re
import os
import time
import sys

from gateway import GateWayServer

from py4j.java_gateway import GatewayParameters


class IXAPosTagger:

    def __init__(self):

#        print "calling constructor"

        # launches java gateway server.
        GateWayServer.launch_gateway()

#        print "attempting to connect to py4j gateway"
#        time.sleep(30)

        self.gateway = JavaGateway(gateway_parameters=GatewayParameters(port=5007), eager_load=True)

        self.tagger = self.gateway.entry_point.getIXAPosTagger()

    def tag(self, naf_tagged_doc):

        if 'NAF' not in naf_tagged_doc:
            exit("text needs to be processed by ixa tokenizer first")

        return self.tagger.tag(naf_tagged_doc)

if __name__ == "__main__":
    pass
