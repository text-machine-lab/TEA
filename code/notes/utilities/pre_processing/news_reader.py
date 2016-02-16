
import subprocess
import os
import re
import sys
import time

xml_utilities_path = os.environ["TEA_PATH"] + "/code/notes/utilities"
sys.path.insert(0, xml_utilities_path)

import xml_utilities
import atexit

import py4j_newsreader.tok
import py4j_newsreader.pos
import py4j_newsreader.parse
import py4j_newsreader.ner

srl = None

class NewsReader(object):

    def __init__(self):

        # newsreader pipeline objects
        print "creating tokenizer..."
        self.newsreader_tok = py4j_newsreader.tok.IXATokenizer()

        print "creating pos tagger..."
        self.newsreader_pos = py4j_newsreader.pos.IXAPosTagger()

        print "creating parser..."
        self.newsreader_parse = py4j_newsreader.parse.IXAParser()

        print "creating ner tagger..."
        self.newsreader_ner = py4j_newsreader.ner.IXANerTagger()
        pass


    def pre_process(self, text):

        """
        the idea behind is to left newsreader do its thing. it uses this formatting called NAF formatting
        that is designed to be this universal markup used by all of the ixa-pipes used in the project.
        """

        tokenized_text  = self.newsreader_tok.tokenize(text)
        pos_tagged_text = self.newsreader_pos.tag(tokenized_text)
        ner_tagged_text = self.newsreader_ner.tag(pos_tagged_text)
        constituency_parsed_text = self.newsreader_parse.parse(ner_tagged_text)
        srl_text = srl.parse_dependencies(constituency_parsed_text)
        coref_tagged_text = _coreference_tag(srl_text)

        # TODO: add more processing steps
        naf_marked_up_text = coref_tagged_text

        return naf_marked_up_text

def _tokenize(text):
    """ takes in path to a file and then tokenizes it """
    tok = subprocess.Popen(["java",
                            "-jar",
                            os.environ["TEA_PATH"] + "/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-tok-1.8.2.jar",
                            "tok",
                            "-l",       # language
                            "en"],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    output, _ = tok.communicate(text)

    return output

def _pos_tag(naf_tokenized_text):

    tag = subprocess.Popen(["java",
                            "-jar",
                            os.environ["TEA_PATH"] + "/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-pos-1.4.1.jar",
                            "tag",
                            "-m",
                            os.environ["TEA_PATH"] + "/code/notes/NewsReader/models/pos-models-1.4.0/en/en-maxent-100-c5-baseline-dict-penn.bin"],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    output, _ = tag.communicate(naf_tokenized_text)

    return output

def _ner_tag(naf_pos_tagged_text):

    tag = subprocess.Popen(["java",
                            "-jar",
                            os.environ["TEA_PATH"] + "/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-nerc-1.5.2.jar",
                            "tag",
                            "-m",
                            os.environ["TEA_PATH"] + "/code/notes/NewsReader/models/nerc-models-1.5.0/nerc-models-1.5.0/en/conll03/en-91-19-conll03.bin"],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    output, _ = tag.communicate(naf_pos_tagged_text)

    return output

def _constituency_parse(naf_tokenized_pos_tagged_text):

    parse = subprocess.Popen(["java",
                              "-jar",
                              os.environ["TEA_PATH"] + "/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-parse-1.1.0.jar",
                              "parse",
                              "-m",
                              os.environ["TEA_PATH"] + "/code/notes/NewsReader/models/parse-models/en-parser-chunking.bin"],
                              stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE)

    output, _ = parse.communicate(naf_tokenized_pos_tagged_text)

    return output

def _coreference_tag(naf_constituency_parsed_text):

    tag = subprocess.Popen(["python2.7",
                            "-m",
                            "corefgraph.process.file",
                            "--reader",
                            "NAF",
                            "--language",
                            "en"],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    output, _ = tag.communicate(naf_constituency_parsed_text)

    filtered_output = ""

    # coref graph adds some garbage lines at the top of the file. Remove everything before the first opening tag.
    for i, char in enumerate(output):
        if char == '<':
            filtered_output = output[i:]
            break

    return filtered_output

class SRL():

    server = None

    def __init__(self):

        # launching server...
        SRL.server = SRLServer()
        SRL.server.launch_server()

    def parse_dependencies(self, naf_tokenized_pos_tagged_text):
        return SRLClient.parse_dependencies(naf_tokenized_pos_tagged_text)

    # TODO: use atexit.

    @staticmethod
    @atexit.register
    def close_server():
        if SRL.server is not None:
            SRL.server.kill_server()


class SRLClient():

    tries = 0

    @staticmethod
    def parse_dependencies(naf_tokenized_pos_tagged_text):

        output = None

        init_time = time.time()

        while True:

            srl = subprocess.Popen(["java",
                                    "-cp",
                                    os.environ["TEA_PATH"] + "/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-srl/IXA-EHU-srl/target/IXA-EHU-srl-3.0.jar",
                                    "ixa.srl.SRLClient",
                                    "en"],
                                    stdout=subprocess.PIPE,
                                    stdin=subprocess.PIPE)

            if time.time() - init_time > 60:
                exit("couldn't get a connection to the srl server")

            _output, _ = srl.communicate(naf_tokenized_pos_tagged_text)

            if _output == "":
                time.sleep(5)
                continue
            else:
                output = _output
                break

        return output


class SRLServer():

    """ will execute srl server """
    def __init__(self):
        self.s = None

    def launch_server(self):

        self.s = subprocess.Popen(["java",
                                   "-cp",
                                   os.environ["TEA_PATH"] + "/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-srl/IXA-EHU-srl/target/IXA-EHU-srl-3.0.jar",
                                   "ixa.srl.SRLServer",
                                    "en"])

    def kill_server(self):

        if self.s is not None:
            self.s.kill()

srl = SRL()

if __name__ == "__main__":

    nr = NewsReader()

    print nr.pre_process("hello world!")

    print nr.pre_process("this is a sentence!")

    pass

# EOF

