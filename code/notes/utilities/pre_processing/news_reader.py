
import subprocess
import os
import re
import sys

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
    constituency_parsed_text = _constituencey_parse(pos_tagged_text)

    # TODO: add more processing steps
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

def _constituencey_parse(naf_tokenized_pos_tagged_text):

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


if __name__ == "__main__":
    print pre_process(open("test.txt", "rb").read())
    pass
# EOF

