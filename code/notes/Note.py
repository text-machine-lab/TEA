
import os
import nltk.data

from utilities.note_utils import valid_path
from utilities.pre_processing.pre_processing import pre_process

from utilities.xml_utilities import write_root_to_file
from utilities.xml_utilities import get_root
from utilities.timeml_utilities import set_text_element
from utilities.timeml_utilities import annotate_text_element


class Note(object):


    def __init__(self, n_path, annotated_n_path=None, debug=False):

        self.debug = debug

        if self.debug: print "Note class: calling __init__"

        # will terminate
        self._set_note_path(n_path, annotated_n_path)


    def __del__(self):

        if self.debug: print "Note class: calling destructor"


    def _set_note_path(self, n_path, annotated_n_path):

        if self.debug: print "Note class: setting note path"

        valid_path(n_path)
        if annotated_n_path is not None:
            valid_path(annotated_n_path)
        self.note_path = n_path
        self.annotated_note_path = annotated_n_path


    def _read(self):

        return open(self.note_path, "rb").read()

    def write(self, timexEvemtLabels, tlinkLabels, idPairs, offsets):
        '''
        Note::write()

        Purpose: add annotations this notes tml file and write new xml tree to a .tml file in the output folder.

        params:
            timexEvemtLabels: list of dictionaries of labels for timex and events. 
            tlinkLabels: list labels for tlink relations
            idPairs: list of pairs of eid or tid that have a one to one correspondance with the tlinkLabels
            offsets: list of offsets tuples used to locate events and timexes specified by the label lists. Have one to one correspondance with both lists of labels.
        '''
        #TODO: create output directory if it does not exist

        root = get_root(self.note_path)

        length = len(offsets)

        # start at back of document to preserve offsets until they are used
        for i in range(1, length):
            index = length - i
            
            if(timexEvemtLabels[index]["entity_label"][0] == "B"):
                start = offsets[index][0]
                end = offsets[index][1]

                #grab any IN tokens and add them to the tag text
                for j in range (1, i):
                    if(timexEvemtLabels[index + j]["entity_label"][0] == "I"):
                        end = offsets[index + j]["entity_label"][1]
                    else:
                        break

                if timexEvemtLabels[index]["entity_type"] == "TIMEX3":
                    annotated_text = annotate_text_element(root, "TIMEX3", start, end, {"tid": timexEvemtLabels[index]["entity_id"], "type":timexEvemtLabels[index]["entity_label"][2:]})
                else:
                    annotated_text = annotate_text_element(root, "EVENT", start, end, {"eid": timexEvemtLabels[index]["entity_id"], "class":timexEvemtLabels[index]["entity_label"][2:]})

                set_text_element(root, annotated_text)

        # tlinkLength = len(tlinkLabels)

        # for i in range(1, tlinkLength):

            
            

        # skip last 9 characters to remove .TE3input suffix
        path = os.environ['TEA_PATH'] + '/output/' + self.note_path.split('/')[-1][:-9]

        write_root_to_file(root, path)

    def _process(self):

        """ setenenize and tokenize text """
        pass

if __name__ == "__main__":
    print "nothing to do"


