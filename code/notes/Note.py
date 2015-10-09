
from note_utils import valid_path

class Note:

    def __init__(self, n_path=None, debug=False):

        self.debug = debug

        if self.debug: print "Note class: calling __init__"

        # will terminate
        self.set_note_path(n_path)

    def __del__(self):

        if self.debug: print "Note class: calling destructor"

        self.set_note_path(None)

    def set_note_path(self, n_path):

        if self.debug: print "Note class: setting note path"

        if n_path is not None:
            valid_path(n_path)
        self.note_path = n_path

if __name__ == "__main__":
    print "nothing to do"

