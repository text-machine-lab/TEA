
import nltk.data

from utilities.note_utils import valid_path
from utilities.note_utils import load_sentenizer

from nltk.tokenize import word_tokenize

class Note(object):


    sentenizer = load_sentenizer()


    def __init__(self, n_path, debug=False):

        self.debug = debug

        if self.debug: print "Note class: calling __init__"

        # will terminate
        self._set_note_path(n_path)
        self.tokenized_data = None

    def __del__(self):

        if self.debug: print "Note class: calling destructor"


    def _set_note_path(self, n_path):

        if self.debug: print "Note class: setting note path"

        valid_path(n_path)
        self.note_path = n_path


    def _read(self):

        return open(self.note_path, "rb").read()


    def _process(self):

        """ setenenize and tokenize text """

        data = self._read()

        sentenized_data = Note._sentenize(data)
        tokenized_data  = Note._tokenize(sentenized_data)

        self.tokenized_data = tokenized_data

        return self.tokenized_data


    @staticmethod
    def _sentenize(text):

        return Note.sentenizer.tokenize(text)


    @staticmethod
    def _tokenize(text):

        lines = []

        for line in text:
            lines.append(word_tokenize(line))

        return lines


if __name__ == "__main__":
    print "nothing to do"


