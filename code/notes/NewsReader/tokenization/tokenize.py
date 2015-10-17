
import subprocess
import os
import re

"/code/notes/NewsReader/tokenization/ixa-pipes-1.1.0-jars/ixa-pipe-tok-1.8.2.jar"

class Tokenizer:

    def __init__(self):
        pass

    @staticmethod
    def tokenizeFile(fPath):
        # TODO: verify fpath
        """ takes in path to a file and then tokenizes it """
        tok = subprocess.Popen(["java",
                                "-jar",
                                os.environ["TEA_PATH"] + "/code/notes/NewsReader/tokenization/ixa-pipes-1.1.0-jars/ixa-pipe-tok-1.8.2.jar",
                                "tok",
                                "-l",       # language
                                "en",
                                "-o",       # output formatting
                                "conll"],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)

        output, _ = tok.communicate(open(fPath, "rb").read())

        return Tokenizer._processOutput(output)

    @staticmethod
    def _processOutput(conllText):
        """ process conll output into list of list of tokens """

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

if __name__ == "__main__":
    print Tokenizer.tokenizeFile("test.txt")
    pass
# EOF

