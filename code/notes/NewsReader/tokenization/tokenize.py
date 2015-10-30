
import subprocess
import os
import re
import sys

xml_utilities_path = os.environ["TEA_PATH"] + "/code/notes/utilities"
sys.path.insert(0, xml_utilities_path)

import xml_utilities

class Tokenizer:

    def __init__(self):
        pass

    @staticmethod
    def tokenize(text):
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

        return Tokenizer._process_output(output)

    @staticmethod
    def _process_output(ixa_tok_output):

        root = xml_utilities.get_root_from_str(ixa_tok_output)

        """
        conllText = re.sub("\*\<P\>\*", "\n", conllText)

        conllText = conllText.split("\n")

        groupings = []

        grouping = []

        for line in conllText:

            if (line == "" and len(grouping) > 0):
                groupings.append(grouping)
                grouping = []

            if line != "":
                grouping.append(line)

        # just in case.
        groupings = [grouping for grouping in groupings if len(grouping) > 0]

        return groupings
        """


if __name__ == "__main__":
    print Tokenizer.tokenize(open("test.txt", "rb").read())
    pass
# EOF

