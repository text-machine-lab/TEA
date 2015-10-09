
import os

if 'TEA_PATH' not in os.environ:
    exit("please defined TEA_PATH, the path of the direct path of the code folder")

import sys

sys.path.insert(0, os.path.join(os.environ['TEA_PATH'], "code/features"))

from Note import Note

from utilities.timeml_utilities import get_raw_text
from utilities.timeml_utilities import get_tagged_entities

from Features import Features

class TimeNote(Note, Features):


    def __init__(self, timeml_note):

        print "called TimeNote constructor"
        _Note = Note.__init__(self, timeml_note)
        _Features = Features.__init__(self)

        self.tokenized_data = None


    def _process(self):

        self.note_path

        data = get_raw_text(self.note_path)

        sentenized_data = TimeNote._sentenize(data)
        tokenized_data = TimeNote._tokenize(sentenized_data)

        self.tokenized_data = tokenized_data

        return self.tokenized_data


    def vectorize(self):

        self._process()

        return self.get_features_vect()


if __name__ == "__main__":
    print "nothing to do"

