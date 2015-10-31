
import os

if 'TEA_PATH' not in os.environ:
    exit("please defined TEA_PATH, the path of the direct path of the code folder")

import sys

sys.path.insert(0, os.path.join(os.environ['TEA_PATH'], "code/features"))

from Note import Note

from utilities.timeml_utilities import get_text
from utilities.timeml_utilities import get_tagged_entities
from utilities.timeml_utilities import get_text_element
from utilities.xml_utilities import get_raw_text

from utilities.pre_processing import pre_processing

from Features import Features

class TimeNote(Note, Features):


    def __init__(self, timeml_note, annotated_timeml):

        print "called TimeNote constructor"
        _Note = Note.__init__(self, timeml_note)
        _Features = Features.__init__(self)

        self.tokenized_data = None

        self._read(timeml_note, annotated_timeml)


    def _read(self, timeml_note, annotated_timeml=None):

        """
        ideas for chunking:

            use offsets and replace each tagged entity with TIMEMLENTITY

            tokenize the body and then tokenize each of the tagged entities

            and substitute where TIMEMLENTITY occurs with my tokenized entities.

            problem:

                during predicting how do i merge labeled stuff together?

                will have chunkings:

                'he can't run that fast'

                [ ['he', 'ca', 'n't', 'run', 'that', 'fast'], .... ]

                [ ['O',  'B',   'I',   'O',   'O',    'O'], ...]

                need to figure out offsets to label when finished

                will have some global index into overall doc

                will to mainpulate some sub offsets when determining if there is a match somewhere.

        """


        tokenized_text = self._process()

#        print tokenized_text

        exit()

        tagged_entities = get_tagged_entities(annotated_timeml)

        raw_text_element = get_text_element(timeml_note)
        raw_text = get_raw_text(raw_text_element)

        labeled_text_element = get_text_element(annotated_timeml)
        labeled_text = get_raw_text(labeled_text_element)

        raw_text = raw_text.strip("<TEXT>\n")
        raw_text = raw_text.strip("<\/TEXT>")

        labeled_text = labeled_text.strip("<TEXT>\n")
        labeled_text = labeled_text.strip("<\/TEXT>")

        raw_index = 0
        labeled_index = 0

        raw_char_offset = 0
        labeled_char_offset = 0

        # should we count?
        count_raw = True
        count_labeled = True

        test1 = ""
        test2 = ""

        start_count = 0
        end_count = 0

        offsets = []

        tagged_element = None

        while raw_index < len(raw_text) or labeled_index < len(labeled_text):

            if raw_index < len(raw_text):
                if count_raw is True:
                    raw_char_offset += 1
                    test1 += raw_text[raw_index]
                raw_index += 1

            if labeled_index < len(labeled_text):

                if labeled_text[labeled_index:labeled_index+1] == '<' and labeled_text[labeled_index:labeled_index+2] != '</':

                    tagged_element = tagged_entities.pop(0)

                    count_labeled = False
                    start_count += 1

                elif labeled_text[labeled_index:labeled_index+2] == '</':
                    count_labeled = False
                    start_count += 1

                if labeled_text[labeled_index:labeled_index+1] == ">":

                    if tagged_element != None:

                        start = labeled_char_offset
                        end   = labeled_char_offset+len(tagged_element.text)

                        offsets.append({"span":(start, end), "token":tagged_element.text})
                        assert raw_text[start:end] == tagged_element.text, "\'{}\' != \'{}\'".format( raw_text[start:end], tagged_element.text)
                        tagged_element = None

                    end_count += 1
                    count_labeled = True

                    labeled_index += 1
                    continue

                if count_labeled is True:
                    labeled_char_offset += 1
                    test2 += labeled_text[labeled_index]

                labeled_index += 1

        assert start_count == end_count, "{} != {}".format(start_count, end_count)
        assert raw_index == len(raw_text) and labeled_index == len(labeled_text)
        assert raw_char_offset == labeled_char_offset
        assert len(tagged_entities) == 0
        assert tagged_element is None

        print offsets

        if annotated_timeml is not None:

            # file is given
            pass


    def _process(self):

        self.note_path

        data = get_text(self.note_path)

        self.tokenized_data = pre_processing.pre_process(data)

        return self.tokenized_data


    def vectorize(self):

        self._process()

        return self.get_features_vect()


if __name__ == "__main__":

    t = TimeNote("APW19980219.0476.tml.TE3input", "APW19980219.0476.tml")

    print "nothing to do"




