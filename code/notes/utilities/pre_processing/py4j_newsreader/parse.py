

from subprocess import Popen, PIPE, STDOUT

from py4j.java_collections import JavaArray
from py4j.java_gateway import JavaGateway

import re
import os
import time
import sys

from gateway import GateWayServer

import tok
import pos
import ner as nerc

class IXAParser:

    def __init__(self):

        print "calling constructor"

        # launches java gateway server.
        GateWayServer.launch_gateway()

        print "attempting to connect to py4j gateway"
#        time.sleep(30)

        self.gateway = JavaGateway(eager_load=True)

        self.parser = self.gateway.entry_point.getIXAParser()

    def parse(self, naf_tagged_doc):

        if 'NAF' not in naf_tagged_doc:
            exit("text needs to be processed by ixa tokenizer first")

        print "tagging..."

        return self.parser.parse(naf_tagged_doc)

if __name__ == "__main__":

    t = tok.IXATokenizer()
    p = pos.IXAPosTagger()
    ner = nerc.IXANerTagger()
    parser = IXAParser()

    tokenized_text = t.tokenize("hello world")

    tokenized_text

    pos_tagged_doc = p.tag(tokenized_text)

    ner_tagged_doc = ner.tag(pos_tagged_doc)

    print parser.parse(ner_tagged_doc)

    print parser.parse(ner_tagged_doc)
