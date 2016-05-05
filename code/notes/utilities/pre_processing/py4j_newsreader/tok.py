"""Allows newsreader tokenization component to be accessed by python.
"""

from subprocess import Popen, PIPE, STDOUT

from py4j.java_collections import JavaArray
from py4j.java_gateway import JavaGateway
from py4j.protocol import Py4JNetworkError

import re
import os
import time
import sys

from .gateway import GateWayServer

class IXATokenizer:

    def __init__(self):

        print("calling constructor")

        #launches java gateway server.
        GateWayServer.launch_gateway()

        print("attempting to connect to py4j gateway")

        self.gateway = JavaGateway()

        init_time = time.time()

        self.tokenizer = None

        while True:

            if time.time() - init_time > 600:
                exit("couldn't get py4j server running")

            try:

                self.tokenizer = self.gateway.entry_point.getIXATokenizer()

                break

            except Py4JNetworkError:

                time.sleep(60)

                continue


    def tokenize(self, text):

        return self.tokenizer.tokenize(text)

if __name__ == "__main__":
    pass
