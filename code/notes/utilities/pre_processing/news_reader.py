
import subprocess
import os
import re
import sys
import time

xml_utilities_path = os.environ["TEA_PATH"] + "/code/notes/utilities"
sys.path.insert(0, xml_utilities_path)

import xml_utilities

def pre_process(text):

    """
    the idea behind is to left newsreader do its thing. it uses this formatting called NAF formatting
    that is designed to be this universal markup used by all of the ixa-pipes used in the project.
    """
    tokenized_text = _tokenize(text)
    pos_tagged_text = _pos_tag(tokenized_text)
    constituency_parsed_text = _constituency_parse(pos_tagged_text)

    # TODO: move this
#    srl = SRL()

#    srl.launch_server()
#    srl_text = srl.parse_dependencies(constituency_parsed_text)

    # TODO: add more processing steps
#    naf_marked_up_text = srl_text

#    srl.close_server()

    naf_marked_up_text = constituency_parsed_text

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


class SRL():

    def __init__(self):

        # launching server...
        self.server = SRLServer()

    def parse_dependencies(self, naf_tokenized_pos_tagged_text):
        return SRLClient.parse_dependencies(naf_tokenized_pos_tagged_text)


    def launch_server(self):
        self.server.launch_server()


    # TODO: use atexit.
    def close_server(self):
        # EXPLICITELY CALL THIS. python doesn't guarantee __del__ is called.
        self.server.kill_server()


class SRLClient():

    tries = 0

    @staticmethod
    def parse_dependencies(naf_tokenized_pos_tagged_text):

        output = None

        while True:

            if SRLClient.tries > 5:
                exit("cannot get srl output. srl server is not running")

            srl = subprocess.Popen(["java",
                                    "-cp",
                                    os.environ["TEA_PATH"] + "/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-srl/ixa-pipe-srl/IXA-EHU-srl/target/IXA-EHU-srl-3.0.jar",
                                    "ixa.srl.SRLClient",
                                    "en"],
                                    stdout=subprocess.PIPE,
                                    stdin=subprocess.PIPE)

            _output, _ = srl.communicate(naf_tokenized_pos_tagged_text)

            if _output == "":
                print "no output when calling srl. trying again."
                SRLClient.tries += 1
                print "sleeping for 1m to wait for server to load..."
                time.sleep(60)
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

        with open(os.devnull, 'w') as fp:
            self.s = subprocess.Popen(["java",
                                       "-cp",
                                       os.environ["TEA_PATH"] + "/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-srl/ixa-pipe-srl/IXA-EHU-srl/target/IXA-EHU-srl-3.0.jar",
                                       "ixa.srl.SRLServer",
                                        "en"],
                                        stdout=fp)

    def kill_server(self):

        if self.s is not None:
            self.s.kill()

    def get_pid(self):

        if self.s is not None:
            return self.s.pid


if __name__ == "__main__":

    pre_process("i ate the bones")

# EOF

