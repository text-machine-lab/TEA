
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

    # TODO: code is pretty bad. I will refactor it later.

    def __init__(self, timeml_note_path, annotated_timeml_path=None):

        print "called TimeNote constructor"
        _Note = Note.__init__(self, timeml_note_path, annotated_timeml_path)
        _Features = Features.__init__(self)


    def get_labeled_timex_entities(self):

        return self._process(self.note_path, self.annotated_note_path, "TIMEX3")


    def get_labeled_tlinked_entities(self):

        # TODO: implement. this is going to require different logic than just reading
        #       because the taggings are annotated outside of the text body.
        pass


    def get_labeled_event_entities(self):

        return self._process(self.note_path, self.annotated_note_path, "EVENT")


    def write(self):
        # TODO: will do after we have results to display.
        pass


    def _process(self, timeml_note, annotated_timeml=None, entities_to_read=None):

        """
            _read gets the tokenized representation of a timeml note and sets
            the appropriate IOB labeling to only the specific entitie to read

            the logic behind this is that we will only want to markup
            specific timeml entitiy types in each pass, we can just disregard all the
            other taggings.
        """

        if entities_to_read is not None and entities_to_read not in ("TIMEX3", "EVENT"):
            exit("ERROR: invalid entity type to filter by")

        data = get_text(self.note_path)

        tokenized_text = pre_processing.pre_process(data)

        if annotated_timeml is not None:

            tagged_entities = get_tagged_entities(annotated_timeml)
            _tagged_entities = get_tagged_entities(annotated_timeml)

            raw_text_element = get_text_element(timeml_note)
            raw_text = get_raw_text(raw_text_element)

            labeled_text_element = get_text_element(annotated_timeml)
            labeled_text = get_raw_text(labeled_text_element)

            # TODO: cleanup
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

            text1 = ""
            text2 = ""

            start_count = 0
            end_count = 0

            offsets = {}

            tagged_element = None

            while raw_index < len(raw_text) or labeled_index < len(labeled_text):

                if raw_index < len(raw_text):
                    if count_raw is True:
                        raw_char_offset += 1
                        text1 += raw_text[raw_index]
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
                            end   = labeled_char_offset+len(tagged_element.text) - 1

                            # spans should be unique?
                            offsets[(start, end)] = {"tagged_xml_element":tagged_element, "text":tagged_element.text}
                            assert raw_text[start:end + 1] == tagged_element.text, "\'{}\' != \'{}\'".format( raw_text[start:end + 1], tagged_element.text)
                            tagged_element = None

                        end_count += 1
                        count_labeled = True

                        labeled_index += 1
                        continue

                    if count_labeled is True:
                        labeled_char_offset += 1
                        text2 += labeled_text[labeled_index]

                    labeled_index += 1

            # lots of checks!
            assert text1 == text2
            assert start_count == end_count, "{} != {}".format(start_count, end_count)
            assert raw_index == len(raw_text) and labeled_index == len(labeled_text)
            assert raw_char_offset == labeled_char_offset
            assert len(tagged_entities) == 0
            assert tagged_element is None
            assert len(offsets) == len(_tagged_entities)


        # need to create a list of tokens
        processed_data = []

        start_tokens = 0

        for sentence_num in sorted(tokenized_text.keys()):

            # list of dicts
            sentence = tokenized_text[sentence_num]

            tmp = []

            for token in sentence:


                if annotated_timeml is not None:
                    # set proper iob label to token

                    # TODO: make a more thorough test of correctness
                    label = TimeNote.get_iob_label(token, offsets, entities_to_read)

                    token.update({"IOB_label":label})

                tmp.append(token)

            processed_data.append(tmp)

        return processed_data


    def vectorize(self):

        self._process()

        return self.get_features_vect()


    @staticmethod
    def get_iob_label(token, offsets, type):

        # NOTE: never call this directly. input is tested within _read

        tok_span = (token["start_offset"], token["end_offset"])

        label = 'O'

        for span in offsets:

            if TimeNote.same_start_offset(span, tok_span):

                labeled_entity = offsets[span]["tagged_xml_element"]

                if labeled_entity.tag != type and type is not None:
                    break

                label = 'B_' + labeled_entity.tag

                break

            elif TimeNote.subsumes(span, tok_span):

                labeled_entity = offsets[span]["tagged_xml_element"]

                if labeled_entity.tag != type and type is not None:
                    break

                label = 'I_' + labeled_entity.tag

                break

        return label

    @staticmethod
    def same_start_offset(span1, span2):
        """
        doees span1 share the same start offset?
        """
        return span1[0] == span2[0]

    @staticmethod
    def subsumes(span1, span2):
        """
        does span1 subsume span2?
        """
        return span1[0] < span2[0] and span2[1] <= span1[1]


if __name__ == "__main__":

#    TimeNote("APW19980219.0476.tml.TE3input")
#    print TimeNote("APW19980219.0476.tml.TE3input", "APW19980219.0476.tml").get_labeled_timex_entities()
#    print TimeNote("APW19980219.0476.tml.TE3input", "APW19980219.0476.tml").get_labeled_event_entities()
    print "nothing to do"




