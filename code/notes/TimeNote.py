
import os
import itertools

if 'TEA_PATH' not in os.environ:
    exit("please defined TEA_PATH, the path of the direct path of the code folder")

import sys
import re

sys.path.insert(0, os.path.join(os.environ['TEA_PATH'], "code/features"))

from Note import Note

from utilities.timeml_utilities import get_text
from utilities.timeml_utilities import get_tagged_entities
from utilities.timeml_utilities import get_text_element
from utilities.timeml_utilities import get_tlinks
from utilities.timeml_utilities import get_make_instances
from utilities.timeml_utilities import get_doctime_timex

from utilities.xml_utilities import get_raw_text
from utilities.pre_processing import pre_processing

from Features import Features

import copy

class TimeNote(Note, Features):

    # TODO: code is very inefficient. refactor to make processing faster and less redundant.

    def __init__(self, timeml_note_path, annotated_timeml_path=None):

        print "called TimeNote constructor"
        _Note = Note.__init__(self, timeml_note_path, annotated_timeml_path)
        _Features = Features.__init__(self)

        self._process(timeml_note_path, annotated_timeml_path)


    def get_timex_iob_labels(self):
        return self.filter_iob_by_type('TIMEX3')

    def get_event_iob_labels(self):
        return self.filter_iob_by_type('EVENT')

    def filter_iob_by_type(self, entity_type):
        assert entity_type in ['EVENT', 'TIMEX3']

        filtered_iob_labels = copy.deepcopy(self.iob_tagged_data)

        for line in filtered_iob_labels:
            for token in line:
                if token["entity_type"] != entity_type:
                    token['IOB_label'] = 'O'

        return filtered_iob_labels

    def get_tlinked_entities(self):

        tagged_data = copy.deepcopy(self.iob_tagged_data)

        t_links = None

        if self.annotated_note_path is not None:
            t_links = get_tlinks(self.annotated_note_path)
            make_instances = get_make_instances(self.annotated_note_path)
        else:
            exit("cannot call get_labeled_tlinked_entities with a annotated_note_path not set")

        temporal_relations = {}

        eiid_to_eid = {}

        for instance in make_instances:
            eiid_to_eid[instance.attrib["eiid"]] = instance.attrib["eventID"]

        gold_tlink_pairs = set()

        for t in t_links:

            link = {}

            # source
            if "eventInstanceID" in t.attrib:
                src_id = eiid_to_eid[t.attrib["eventInstanceID"]]
            else:
                src_id = t.attrib["timeID"]

            # target
            if "relatedToEventInstance" in t.attrib:
                target_id = eiid_to_eid[t.attrib["relatedToEventInstance"]]
            else:
                target_id = t.attrib["relatedToTime"]

            tmp_dict = {"target_id":target_id, "rel_type":t.attrib["relType"], "lid":t.attrib["lid"]}

            gold_tlink_pairs.add((src_id, target_id))

            if src_id in temporal_relations:


                # this would mean the same src id will map to same target with different relation type.
                # not possible.
                assert tmp_dict not in temporal_relations[src_id]

                temporal_relations[src_id].append(tmp_dict)

            else:
                temporal_relations[src_id] = [tmp_dict]

        assert( len(gold_tlink_pairs) == len(t_links) )

        event_ids = set()
        timex_ids = set()

        chunks = []
        chunk = []

        # get tagged entities and group into a list
        for line in tagged_data:

            for token in line:

                # start of entity
                if re.search('^B_', token["IOB_label"]):

                    if token["entity_type"] == "EVENT":
                        event_ids.add(token["entity_id"])
                    else:
                        timex_ids.add(token["entity_id"])

                    if len(chunk) != 0:
                        chunks.append(chunk)
                        chunk = [token]
                    else:
                        chunk.append(token)

                elif re.search('^I_', token["IOB_label"]):
                    chunk.append(token)
                else:
                    pass

        if len(chunk) != 0:
            chunks.append(chunk)
        chunk = []

        assert len(event_ids.union(timex_ids)) == len(chunks)

        id_chunk_map = {}

        # are all chunks the same entity id?
        for chunk in chunks:

            start_entity_id = chunk[0]["entity_id"]

            for entity in chunk:

                assert entity["entity_id"] == start_entity_id

                if start_entity_id not in id_chunk_map:
                    id_chunk_map[start_entity_id] = [entity]

                else:
                    id_chunk_map[start_entity_id].append(entity)

        assert len(id_chunk_map.keys()) == len(event_ids.union(timex_ids))

        # TODO: hacky, revise this.
        # add doc time. this is a timex.
        doctime = get_doctime_timex(self.note_path)

        doctime_id = doctime.attrib["tid"]

        doctime_dict = {}

        # create dict representation of doctime timex
        for attrib in doctime.attrib:

            doctime_dict[attrib] = doctime.attrib[attrib]

        id_chunk_map[doctime_id] = doctime_dict

        timex_ids.add(doctime_id)

        # cartesian product of entity pairs
        entity_pairs = filter(lambda t: t[0] != t[1], list(itertools.product(event_ids, timex_ids)) +\
                                                      list(itertools.product(timex_ids, event_ids)) +\
                                                      list(itertools.product(event_ids, event_ids)) +\
                                                      list(itertools.product(timex_ids, timex_ids)))

        entity_pairs = set(entity_pairs)

        relation_count = 0

        pairs_to_link = []

        for pair in entity_pairs:

            src_id = pair[0]
            target_id = pair[1]

            if src_id in temporal_relations:

                relation_found = False

                for target_entity in temporal_relations[src_id]:

                    if target_id == target_entity["target_id"]:

                        relation_count += 1

                        # need to assign relation to each pairing if there exists one otherwise set 'none'
                        pairs_to_link.append({"src_entity":id_chunk_map[src_id], "target_entity":id_chunk_map[target_id], "rel_type":target_entity["rel_type"], "tlink_id":target_entity["lid"]})

                    else:

                        pairs_to_link.append({"src_entity":id_chunk_map[src_id], "target_entity":id_chunk_map[target_id], "rel_type":'None', "tlink_id":None})

        assert( relation_count == len(t_links) )

        return pairs_to_link


    def _process(self, timeml_note, annotated_timeml=None):

        """
            _read gets the tokenized representation of a timeml note and sets
            the appropriate IOB labeling to only the specific entitie to read

            the logic behind this is that we will only want to markup
            specific timeml entitiy types in each pass, we can just disregard all the
            other taggings.
        """

        data = get_text(timeml_note)

        self.original_text = data

        tokenized_text = pre_processing.pre_process(data)

        self.tokenized_text = copy.deepcopy(tokenized_text)

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

            self.offsets = offsets


        # need to create a list of tokens
        processed_data = []

        start_tokens = 0

        self.sentence_lengths = []

        for sentence_num in sorted(tokenized_text.keys()):

            # list of dicts
            sentence = tokenized_text[sentence_num]

            sentence_len = len(sentence)

            self.sentence_lengths.append(sentence_len)

            tmp = []

            for token in sentence:


                if annotated_timeml is not None:
                    # set proper iob label to token

                    label, entity_type, entity_id = TimeNote.get_iob_labels(token, offsets)

                    if label is None:
                        assert entity_id is not None

                    assert entity_type in ['EVENT', 'TIMEX3', None]

                    token.update({"IOB_label":label})
                    token.update({"entity_id":entity_id})
                    token.update({"entity_type":entity_type})

                else:
                    token.update({"IOB_label":'O'})
                    token.update({"entity_id":None})
                    token.update({"entity_type":None})

                tmp.append(token)

            processed_data.append(tmp)

        self.iob_tagged_data = processed_data


    def vectorize(self, entity_filter):

        # TODO: i guess the iob label may notm atter at this stage.

        labels = []
        offsets = []
        tlink_ids = []

        processed_data = None

        if entity_filter == "EVENT":
            processed_data = self.get_event_iob_labels()

        elif entity_filter == "TIMEX3":
            processed_data = self.get_timex_iob_labels()

        elif entity_filter == 'TLINK':
            processed_data = self.get_tlinked_entities()

        else:
            exit("not EVENT, TIMEX3, TLINK")

        if entity_filter in ['EVENT', 'TIMEX3']:

            for line in processed_data:

                for token in line:

                    labels.append(token["IOB_label"])
                    offsets.append((token["start_offset"], token["end_offset"]))

            return self.get_features_vect_tokenized_data(processed_data), labels, offsets

        else:


            # tlinks
            for tlink in processed_data:

                labels.append(tlink["rel_type"])

                tlink_ids.append(tlink["tlink_id"])

            return self.get_features_vect_tlinks(processed_data), labels, tlink_ids

    @staticmethod
    def get_iob_labels(token, offsets):

        # NOTE: never call this directly. input is tested within _read

        tok_span = (token["start_offset"], token["end_offset"])

        label = 'O'
        entity_id = None
        entity_type = None

        for span in offsets:

            if TimeNote.same_start_offset(span, tok_span):

                labeled_entity = offsets[span]["tagged_xml_element"]

                if 'class' in labeled_entity.attrib:
                    label = 'B_' + labeled_entity.attrib["class"]
                else:
                    label = 'B_' + labeled_entity.attrib["type"]

                if 'eid' in labeled_entity.attrib:
                    entity_id = labeled_entity.attrib["eid"]
                else:
                    entity_id = labeled_entity.attrib["tid"]

                entity_type = labeled_entity.tag

                break

            elif TimeNote.subsumes(span, tok_span):

                labeled_entity = offsets[span]["tagged_xml_element"]

                if 'class' in labeled_entity.attrib:
                    label = 'I_' + labeled_entity.attrib["class"]
                else:
                    label = 'I_' + labeled_entity.attrib["type"]

                if 'eid' in labeled_entity.attrib:
                    entity_id = labeled_entity.attrib["eid"]
                else:
                    entity_id = labeled_entity.attrib["tid"]

                entity_type = labeled_entity.tag

                break

        return label, entity_type, entity_id

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

    def get_features_vect_tokenized_data(self, processed_data):

        # TODO: generate feature set

        vectors = []

        for line in processed_data:

            for token in line:

                vectors.append({"dummy":1})

        return vectors

    def get_features_vect_tlinks(self, processed_data):

        vectors = []

        for relation in processed_data:

            vectors.append({'e1_dummy':1, 'e2_dummy':1})

        return vectors

    def create_features_vect_tlinks(self, entity_pairs):

        # TODO: this will be done after training is completed.

        pass


if __name__ == "__main__":

    t =  TimeNote("APW19980219.0476.tml.TE3input", "APW19980219.0476.tml")

    _,_,lids= t.vectorize('TLINK')

    print lids

    print "nothing to do"




