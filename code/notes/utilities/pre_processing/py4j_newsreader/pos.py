

from subprocess import Popen, PIPE, STDOUT

from py4j.java_collections import JavaArray
from py4j.java_gateway import JavaGateway

import re
import os
import time
import sys

from gateway import GateWayServer

class IXAPosTagger:

    def __init__(self):

        print "calling constructor"

        # launches java gateway server.
        GateWayServer.launch_gateway()

        print "attempting to connect to py4j gateway"
#        time.sleep(30)

        self.gateway = JavaGateway(eager_load=True)

        self.tagger = self.gateway.entry_point.getIXAPosTagger()

    def tag(self, naf_tagged_doc):

        if 'NAF' not in naf_tagged_doc:
            exit("text needs to be processed by ixa tokenizer first")

        print "tagging..."

        return self.tagger.tag(naf_tagged_doc)

if __name__ == "__main__":
    pass
