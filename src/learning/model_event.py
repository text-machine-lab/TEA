from src.notes.utilities import timeml_utilities
from src.notes.utilities.xml_utilities import write_root_to_file
from src.notes.TimeNote import TimeNote
from keras.models import model_from_json
from event_network import EventNetwork
from src.learning.word2vec import load_word2vec_binary
from src.notes.utilities.time_norm import get_normalized_time_expressions
from src.notes.utilities.timeml_utilities import get_doctime_timex

import glob
import os
import cPickle
import numpy
import json
numpy.random.seed(1337)

EVENT_POS_TAGS = {'A':'ADJECTIVE', 'N':'NOUN', 'V':'VERB', 'P':'PREP'}


convert_pos_tags = lambda tag: EVENT_POS_TAGS.get(tag[0], 'UNKNOWN')


class EventWriter(object):

    def __init__(self, note, word_vectors=None, NNet=None):
        self.note = note
        self.predicate_tokens = []
        self.event_tokens_network = []
        self.make_instances = []
        self.event_elements = []
        #self.find_predicates()

        if NNet:
            self.find_event_tokens(word_vectors, NNet)



    def find_predicates(self):
        for sent_num in self.note.pre_processed_text:
            predicate_tokens = [tok for tok in self.note.pre_processed_text[sent_num] if tok.get('is_predicate', False)]
            self.predicate_tokens += predicate_tokens

    def find_event_tokens(self, word_vectors, NNet):
        if word_vectors is None:
            word_vectors = load_word2vec_binary(os.environ["TEA_PATH"] + '/GoogleNews-vectors-negative300.bin', verbose=0)

        event_network = EventNetwork(word_vectors=word_vectors)
        data = event_network.get_test_input(self.note)
        print "predicting events..."
        predictions, probs = event_network.predict(NNet, data)
        for i, pred in enumerate(predictions):
            if pred:
                wordID = 'w' + str(i+1)
                token = self.note.id_to_tok[wordID]
                self.event_tokens_network.append(token)

    def tag_predicates(self):
        make_instances = []
        event_elements = []
        index = len(self.predicate_tokens)
        for token in reversed(self.predicate_tokens):
            start = token['char_start_offset']
            end = token['char_end_offset']
            eid = 'e' + str(index)
            event_elements.append({"tag": 'EVENT',
                                   "start_offset": start,
                                   "end_offset": end,
                                   "eid": eid})

            eiid = 'ei' + str(index)
            pos = convert_pos_tags(token['morpho_pos'])
            make_instances.append({'eventID':eid, 'eiid':eiid, 'pos':pos, 'tense':'NONE'}) # no tense info for now

            index -= 1

        self.make_instances = make_instances
        self.event_elements = event_elements

        return event_elements

    def tag_events(self):
        make_instances = []
        event_elements = []
        index = len(self.event_tokens_network)
        for token in reversed(self.event_tokens_network):
            start = token['char_start_offset']
            end = token['char_end_offset']
            eid = 'e' + str(index)
            event_elements.append({"tag": 'EVENT',
                                   "start_offset": start,
                                   "end_offset": end,
                                   "eid": eid})

            eiid = 'ei' + str(index)
            pos = convert_pos_tags(token['morpho_pos'])
            make_instances.append({'eventID':eid, 'eiid':eiid, 'pos':pos, 'tense':'NONE'}) # no tense info for now

            index -= 1

        self.make_instances = make_instances
        self.event_elements = event_elements

        return event_elements

    def tag_text(self, timexLabels, tokens, note):
        timeml_root = timeml_utilities.get_stripped_root(self.note.note_path)

        if self.event_elements:
            event_elements = self.event_elements
        else:
            #event_elements = self.tag_predicates()
            event_elements = self.tag_events()

        if timexLabels:
            timex_elements = tag_timex(timexLabels, tokens, self.note)
        else:
            timex_elements = []

        # not allowing event in timex. timex has a higher priority
        timex_positions = [e['start_offset'] for e in timex_elements]
        event_elements = [item for item in event_elements if item['start_offset'] not in timex_positions]

        tagged_elements = event_elements + timex_elements
        tagged_elements.sort(key=lambda k: k['start_offset'], reverse=True)
        print tagged_elements

        index = len(tagged_elements)
        previous_position = None
        for e in tagged_elements:
            tag = e['tag']
            start = e['start_offset']
            end = e['end_offset']
            if previous_position:
                if start == previous_position[0] or end > previous_position[0]: # overlapped tags are not allowed
                    continue
            if tag == 'EVENT':
                # TODO: find a way to identify 'class'
                annotated_text = timeml_utilities.annotate_text_element(timeml_root, tag, start, end,
                                                                        attributes = {'eid':e['eid'], 'class':'OCCURRENCE'})
            elif tag == 'TIMEX3':
                annotated_text = timeml_utilities.annotate_text_element(timeml_root, tag, start, end,
                                                                        attributes={'tid': e['tid'], 'type':e['type'],
                                                                                    'value': e['value']})

            timeml_root = timeml_utilities.set_text_element(timeml_root, annotated_text)
            index -= 1
            previous_position = (start, end)

        for new_instance in self.make_instances:
            timeml_root = timeml_utilities.annotate_root(timeml_root, 'MAKEINSTANCE', attributes=new_instance)

        return timeml_root

    @staticmethod
    def write_tags(timeml_root, output_path):
        write_root_to_file(timeml_root, output_path)


def tag_timex(timexLabels, tokens, note):

    offsets = note.get_token_char_offsets()
    length = len(offsets)
    timex_elements = []
    doc_time = get_doctime_timex(note.note_path).attrib["value"]

    # # hack so events are detected in next for loop.
    # for label in timexLabels:
    #     if label["entity_label"][0:2] not in ["B_", "I_", "O"] or label["entity_label"] in ["I_STATE", "I_ACTION"]:
    #         label["entity_label"] = "B_" + label["entity_label"]

    # start at back of document to preserve offsets until they are used
    for i in range(1, length + 1):
        index = length - i

        if timexLabels[index]["entity_label"][0:2] == "B_":
            start = offsets[index][0]
            end = offsets[index][1]
            entity_tokens = tokens[index]["token"]

            # grab any IN tokens and add them to the tag text
            for j in range(1, i):

                if timexLabels[index + j]["entity_label"][0:2] == "I_":
                    end = offsets[index + j][1]
                    entity_tokens += ' ' + tokens[index + j]["token"]
                else:
                    break

            if timexLabels[index]["entity_type"] == "TIMEX3":
                # get the time norm value of the time expression
                #timex_value = get_normalized_time_expressions(doc_time, [entity_tokens])

                timex_value = timexLabels[index]["entity_value"]
                print "timex_value", timex_value

                timex_elements.append({"tag": "TIMEX3",
                                       "start_offset": start,
                                       "end_offset": end,
                                       "tid": timexLabels[index]["entity_id"],
                                       "type": timexLabels[index]["entity_label"][2:],
                                       "value": timex_value})

    return timex_elements


def etree_to_dict(t):
    d = {t.tag: map(etree_to_dict, t.iterchildren())}
    d.update(('@' + k, v) for k, v in t.attrib.iteritems())
    d['text'] = t.text
    return d


def evaluate_event_tagging(pred_note, gold_note, *args):

    # need this because old version notes do not have event_ids attribute
    if args:
        gold_event_ids = args[0]
    else:
        gold_event_ids = gold_note.event_ids

    pred_event_wordIDs = set([tuple(pred_note.id_to_wordIDs[x]) for x in pred_note.event_ids])
    gold_event_wordIDs = set([tuple(gold_note.id_to_wordIDs[x]) for x in gold_event_ids])

    true_pos = len(gold_event_wordIDs & pred_event_wordIDs)
    false_pos = len(pred_event_wordIDs - gold_event_wordIDs)
    false_neg = len(gold_event_wordIDs - pred_event_wordIDs)

    return true_pos, false_pos, false_neg


def evaluate_timex_tagging(pred_note, gold_note, *args):

    # need this because old version notes do not have timex_ids attribute
    if args:
        gold_timex_ids = args[0]
    else:
        gold_timex_ids = gold_note.event_ids

    pred_timex_wordIDs = set([tuple(pred_note.id_to_wordIDs[x]) for x in pred_note.timex_ids])
    gold_timex_wordIDs = set([tuple(gold_note.id_to_wordIDs[x]) for x in gold_timex_ids])

    true_pos = len(gold_timex_wordIDs & pred_timex_wordIDs)
    false_pos = len(pred_timex_wordIDs - gold_timex_wordIDs)
    false_neg = len(gold_timex_wordIDs - pred_timex_wordIDs)

    return true_pos, false_pos, false_neg


def evaluate_tagging(annotated_dir, newsreader_dir):

    annotated_files = sorted(glob.glob(os.path.join(annotated_dir,'*.tml')))

    base_names = [os.path.basename(x) for x in annotated_files]
    gold_notes = [os.path.join(newsreader_dir, x[0:x.index(".tml")]+'.parsed.pickle') for x in base_names]

    event_results = numpy.array([0, 0, 0])
    timex_results = numpy.array([0, 0, 0])

    for i, gold_note_file in enumerate(gold_notes):
        event_ids = []
        timex_ids = []
        print "processing file: ", base_names[i]
        pred_file = annotated_files[i]
        pred_note = TimeNote(pred_file, pred_file)
        gold_note = cPickle.load(open(gold_note_file))
        try:
            len(gold_note.event_ids)
            event_ids = gold_note.event_ids
            timex_ids = gold_note.timex_ids
            #print "event ids from gold file:", gold_note.event_ids
        except AttributeError: # old version notes do not have self.event_ids
            print "retrieving event ids and timex ids..."
            id_chunk_map, event_ids, timex_ids, sentence_chunks = gold_note.get_id_chunk_map()

        event_results += numpy.array(evaluate_event_tagging(pred_note, gold_note, event_ids))
        timex_results += numpy.array(evaluate_timex_tagging(pred_note, gold_note, timex_ids))

    eval = {}
    eval['event_precision'] = event_results[0] * 1.0 / (event_results[0] + event_results[1])
    eval['event_recall'] = event_results[0] * 1.0 / (event_results[0] + event_results[2])
    eval['event_f1'] = 2*eval['event_precision']*eval['event_recall'] / (eval['event_precision']+eval['event_recall'])

    eval['timex_precision'] = timex_results[0] * 1.0 / (timex_results[0] + timex_results[1])
    eval['timex_recall'] = timex_results[0] * 1.0 / (timex_results[0] + timex_results[2])
    eval['timex_f1'] = 2*eval['timex_precision']*eval['timex_recall'] / (eval['timex_precision'] + eval['timex_recall'])

    #print eval
    json.dump(eval, open(os.path.join(annotated_dir, 'eval.json'), 'w'))
    return eval



