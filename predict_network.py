"""
Interface to perform predicting of TIMEX, EVENT and TLINK annotations,
using a neural network model for TLINK annotations.
"""

import sys
from code.config import env_paths
import numpy
numpy.random.seed(1337)

# this needs to be set. exit now so user doesn't wait to know.
if env_paths()["PY4J_DIR_PATH"] is None:
    sys.exit("PY4J_DIR_PATH environment variable not specified")

import argparse
import cPickle
import glob
import os

from keras.models import model_from_json

from code.learning.network import Network
from code.notes.TimeNote import TimeNote
from code.learning.time_ref import predict_timex_rel
from code.learning.break_cycle import modify_tlinks

timenote_imported = False


def basename(s):
    if '.tml' in s:
        s = os.path.basename(s[0:s.index(".tml")])
    if 'TE3input' in s:
        s = os.path.basename(s[0:s.index(".TE3input")])
    return s


def main():

    global timenote_imported

    parser = argparse.ArgumentParser()

    parser.add_argument("predict_dir",
                        nargs=1,
                        help="Directory containing test input")

    parser.add_argument("intra_model_path",
                        help="Where trained model for intra-sentence pairs is located")

    parser.add_argument("cross_model_path",
                        help="Where trained model for cross-sentence pairs is located")

    parser.add_argument("dct_model_path",
                        help="Where trained model for events and document creation time is located")

    parser.add_argument("annotation_destination",
                         help="Where annotated files are written")

    parser.add_argument("newsreader_annotations",
                        help="Where newsreader pipeline parsed file objects go")

    # parser.add_argument("--use_gold",
    #                     action='store_true',
    #                     default=False,
    #                     help="Use gold taggings for EVENT and TIMEX")

    # parser.add_argument("--single_pass",
    #                     action='store_true',
    #                     default=False,
    #                     help="Predict using a single pass model")

    parser.add_argument("--evaluate",
                        action='store_true',
                        default=False,
                        help="Use gold data from the given files to produce evaluation metrics")

    args = parser.parse_args()

    annotation_destination = args.annotation_destination
    newsreader_dir = args.newsreader_annotations

    if os.path.isdir(annotation_destination) is False:
        sys.exit("\n\noutput destination does not exist")
    if os.path.isdir(newsreader_dir) is False:
        sys.exit("invalid path for time note dir")

    predict_dir = args.predict_dir[0]

    if os.path.isdir(predict_dir) is False:
        sys.exit("\n\nno output directory exists at set path")

    # pickled_timeml_notes = [os.path.basename(l) for l in glob.glob(newsreader_dir + "/*")]

    if '/*' != args.predict_dir[0][-2:]:
        predict_dir = predict_dir + '/*'

    # get files in directory
    files = glob.glob(predict_dir)

    gold_files = []
    tml_files  = []

    for f in files:
        if f.endswith(".TE3input"): #input file without tlinks
            tml_files.append(f)
        elif f.endswith(".tml"):
            gold_files.append(f)

    gold_files.sort()
    tml_files.sort()
    print "gold_files", gold_files

    # one-to-one pairing of annotated file and un-annotated
    # assert len(gold_files) == len(tml_files)

    network = Network()

    intra_model = model_from_json(open(os.path.join(args.intra_model_path, 'intra', '.arch.json')).read())
    intra_model.load_weights(os.path.join(args.intra_model_path, 'intra', '.weights.h5'))
    cross_model = model_from_json(open(os.path.join(args.cross_model_path, 'cross', '.arch.json')).read())
    cross_model.load_weights(os.path.join(args.cross_model_path, 'cross', '.weights.h5'))
    dct_model = model_from_json(open(os.path.join(args.dct_model_path, 'dct', '.arch.json')).read())
    dct_model.load_weights(os.path.join(args.dct_model_path, 'dct', '.weights.h5'))

    for i, tml in enumerate(gold_files):

        print '\n\nprocessing file {}/{} {}'.format(i + 1,
                                                    len(gold_files),
                                                    tml)
        if os.path.isfile(os.path.join(newsreader_dir, basename(tml) + ".parsed.pickle")):
            tmp_note = cPickle.load(open(os.path.join(newsreader_dir, basename(tml) + ".parsed.pickle"), "rb"))
        else:
            tmp_note = TimeNote(tml, tml)
            cPickle.dump(tmp_note, open(newsreader_dir + "/" + basename(tml) + ".parsed.pickle", "wb"))

        # notes.append(tmp_note)
        notes = [tmp_note] # required to be a list

    # else: # not using gold
    #     if '/*' != args.predict_dir[0][-2:]:
    #         predict_dir = predict_dir + '/*'
    #
    #     # get files in directory
    #     files = glob.glob(predict_dir)
    #
    #     tml_files  = []
    #
    #     for f in files:
    #         if f.endswith(".TE3input"): #input file without tlinks
    #             tml_files.append(f)
    #
    #     tml_files.sort()
    #     print "tml_files", tml_files
    #
    #     tmp_note = None
    #
    #     for i, tml in enumerate(tml_files):
    #         if basename(tml) + ".parsed.pickle" in pickled_timeml_notes:
    #             tmp_note = cPickle.load(open(newsreader_dir + "/" + basename(tml) + ".parsed.pickle", "rb"))
    #         else:
    #             if timenote_imported is False:
    #                 timenote_imported = True
    #             tmp_note = TimeNote(tml, None)
    #             cPickle.dump(tmp_note, open(newsreader_dir + "/" + basename(tml) + ".parsed.pickle", "wb"))
    #
    #         notes.append(tmp_note)

        intra_labels, intra_probs, intra_pair_index = network.single_predict(notes, intra_model, 'intra', predict_prob=True)
        intra_labels, intra_pair_index, intra_scores = network.smart_predict(intra_labels, intra_probs, intra_pair_index, type='str')

        cross_labels, cross_probs, cross_pair_index = network.single_predict(notes, cross_model, 'cross', predict_prob=True)
        cross_labels, cross_pair_index, cross_scores = network.smart_predict(cross_labels, cross_probs, cross_pair_index, type='str')

        timex_labels, timex_pair_index = predict_timex_rel(notes)

        dct_labels, dct_probs, dct_pair_index = network.single_predict(notes, dct_model, 'dct', predict_prob=True)
        dct_labels = network._convert_int_labels_to_str(dct_labels)
        dct_scores = [max(probs) for probs in dct_probs]
        assert len(dct_labels) == len(dct_scores)

        for i, note in enumerate(notes):
            note_id_pairs = []
            note_labels = []
            note_scores = []

            for key in intra_pair_index.keys(): #  {(note_id, (ei, ej)) : index}
                # the dictionary is dynamically changing, so we need to check
                if key not in intra_pair_index:
                    continue
                if key[0] == i:
                    note_id_pairs.append(key[1])
                    note_labels.append(intra_labels[intra_pair_index[key]])
                    note_scores.append(intra_scores[intra_pair_index[key]])
                    intra_pair_index.pop(key)
                    opposite_key = (key[0], (key[1][1], key[1][0]))
                    intra_pair_index.pop(opposite_key)

            for key in cross_pair_index.keys():  # {(note_id, (ei, ej)) : index}
                # the dictionary is dynamically changing, so we need to check
                if key not in cross_pair_index:
                    continue
                if key[0] == i:
                    note_id_pairs.append(key[1])
                    note_labels.append(cross_labels[cross_pair_index[key]])
                    note_scores.append(cross_scores[cross_pair_index[key]])
                    cross_pair_index.pop(key)
                    opposite_key = (key[0], (key[1][1], key[1][0]))
                    cross_pair_index.pop(opposite_key)

            for key in timex_pair_index.keys():  # {(note_id, (t, t)) : index}
                if key[0] == i:
                    note_id_pairs.append(key[1])
                    note_labels.append(timex_labels[timex_pair_index[key]])
                    note_scores.append(1.0) # trust timex tlinks
                    timex_pair_index.pop(key)

            for key in dct_pair_index.keys():  # {(note_id, (ei, t0)) : index}
                if key[0] == i:
                    note_id_pairs.append(key[1])
                    note_labels.append(dct_labels[dct_pair_index[key]])
                    note_scores.append(max(dct_probs[dct_pair_index[key]]))
                    #note_scores.append(0.0)
                    dct_pair_index.pop(key)

            note_labels, note_scores = resolve_coref(note, note_id_pairs, note_labels, note_scores)
            note_labels = modify_tlinks(note_id_pairs, note_labels, note_scores)
            save_predictions(note, note_id_pairs, note_labels, annotation_destination)


    # label_index = 0
    # for note in notes:
    #     #id_pairs = network._extract_path_words(note).keys()
    #     #id_pairs.sort()
    #
    #     n_pairs = len(id_pairs)
    #     note_labels = labels[label_index:label_index+n_pairs]
    #     note_label_nums = network._convert_str_labels_to_int(note_labels)
    #     label_index += n_pairs
    #
    #     processed_entities = {}
    #     used_indexes = []
    #     # for the same entity pairs (regardless of order), only use the best scores
    #     for i, note_label_num in enumerate(note_label_nums):
    #         if max(probs[i]) < 0.2:
    #             continue
    #         if (id_pairs[i][1], id_pairs[i][0]) in processed_entities:
    #             if probs[i][note_label_num] > processed_entities[(id_pairs[i][1], id_pairs[i][0])]:
    #                 used_indexes.append(i)  # reverse order
    #             else:
    #                 used_indexes.append(i - 1) # (e1, e2) and (e2, e1) are next to each other
    #         else:
    #             processed_entities[(id_pairs[i][0], id_pairs[i][1])] = probs[i][note_label_num]
    #
    #     note_labels = [note_labels[x] for x in used_indexes]
    #     used_pairs = [id_pairs[x] for x in used_indexes]
    #
    #     save_predictions(note, used_pairs, note_labels, annotation_destination)

def normalize_scores(scores):
    scores = numpy.array(scores)
    m = numpy.mean(scores)
    s = numpy.std(scores)
    norm = (scores - m) / s
    print "mean and std of scores:", m, s
    return norm


def resolve_coref(note, note_id_pairs, note_labels, note_scores):
    for i, id_pair in enumerate(note_id_pairs):
        if 't0' in id_pair:
            continue
        src_eid, target_eid = id_pair
        src_coref = note.id_to_tok[note.id_to_wordIDs[src_eid][0]].get("coref_chain", "no_coref1")
        target_coref = note.id_to_tok[note.id_to_wordIDs[target_eid][0]].get("coref_chain", "no_coref2")
        if src_coref == target_coref:
            note_labels[i] = 'SIMULTANEOUS'
            print "event coreference found:", src_eid, target_eid
            note_scores[i] = 0.9
    return note_labels, note_scores


def save_predictions(note, id_pairs, note_labels, annotation_destination):
    note_path = os.path.join(annotation_destination, note.note_path.split('/')[-1])
    print "saving predictions in", note_path
    with open(note.annotated_note_path, 'r') as f:
        raw_text = []
        for line in f:
            if '<MAKEINSTANCE' in line:
                break
            elif '<TLINK' in line:
                break
            else:
                raw_text.append(line)

    raw_text = ''.join(raw_text)
    tlinks = []
    makeinstances = []

    for i, (id_pair, note_label) in enumerate(zip(id_pairs, note_labels)):
        if note_label != 'None':
            src_eid, target_eid = id_pair
            if src_eid[0] == 'e' and target_eid[0] == 'e':
                src_eiid = 'ei' + src_eid[1:]
                target_eiid = 'ei' + target_eid[1:]
                lid = 'l' + str(i)
                tlink = '<TLINK eventInstanceID="{}" lid="{}" relType="{}" relatedToEventInstance="{}"/>'.format(
                                                                                                    src_eiid, lid,
                                                                                                    note_label, target_eiid)
                makeinstance = '<MAKEINSTANCE eiid="{}" eventID="{}" pos="UNKNOWN" tense="NONE"/>'.format(src_eiid, src_eid)
                makeinstances.append(makeinstance)
                makeinstance = '<MAKEINSTANCE eiid="{}" eventID="{}" pos="UNKNOWN" tense="NONE"/>'.format(target_eiid, target_eid)
                makeinstances.append(makeinstance)

            elif src_eid[0] == 'e' and target_eid[0] == 't':
                src_eiid = 'ei' + src_eid[1:]
                target_tid = target_eid
                lid = 'l' + str(i)
                tlink = '<TLINK eventInstanceID="{}" lid="{}" relType="{}" relatedToTime="{}"/>'.format(src_eiid, lid,
                                                                                                note_label, target_tid)
                makeinstance = '<MAKEINSTANCE eiid="{}" eventID="{}" pos="UNKNOWN" tense="NONE"/>'.format(src_eiid, src_eid)
                makeinstances.append(makeinstance)

            elif src_eid[0] == 't' and target_eid[0] == 'e':
                src_tid = src_eid
                target_eiid = 'ei' + target_eid[1:]
                lid = 'l' + str(i)
                tlink = '<TLINK timeID="{}" lid="{}" relType="{}" relatedToEventInstance="{}"/>'.format(src_tid, lid,
                                                                                                note_label, target_eiid)
                makeinstance = '<MAKEINSTANCE eiid="{}" eventID="{}" pos="UNKNOWN" tense="NONE"/>'.format(target_eiid, target_eid)
                makeinstances.append(makeinstance)

            elif src_eid[0] == 't' and target_eid[0] == 't':
                src_tid = src_eid
                target_tid = target_eid
                lid = 'l' + str(i)
                tlink = '<TLINK timeID="{}" lid="{}" relType="{}" relatedToTime="{}"/>'.format(src_tid, lid,
                                                                                               note_label, target_tid)
            tlinks.append(tlink)

    raw_text = raw_text.replace('</TimeML>', '')
    makeinstances = sorted(list(set(makeinstances)))
    for makeinstance in makeinstances:
        raw_text += '\n' + makeinstance
    for tlink in sorted(list(set(tlinks))):
        raw_text += '\n' + tlink

    raw_text += '\n</TimeML>'

    with open(note_path, 'w') as f:
        f.write(raw_text)


def process_note(note, labels, del_list, label_index, probs):
    # get entity pairs, offsets, tokens, and event/timex entities
    entities = note.get_tlink_id_pairs()
    offsets = note.get_token_char_offsets()

    # flatten list of tokens
    tokenized_text = note.get_tokenized_text()
    tokens = []
    for line in tokenized_text:
        tokens += tokenized_text[line]

    event_timex_labels = []
    # flatten list of labels
    for label_list in note.get_labels():
        event_timex_labels += label_list

    # sort del_list to be in ascending order, and remove duplicates
    del_list = list(set(del_list))
    del_list.sort()
    del_list.reverse()
    # loop through indicies starting at the back to preserve earlier indexes
    for index in del_list:
        del entities[index]

    note_labels = labels[label_index:label_index + len(entities)]
    note_label_nums = Network()._convert_str_labels_to_int(note_labels)

    processed_entities = {}
    used_indexes = []
    # for the same entity pairs (regardless of order), only use the best scores
    for i, note_label_num in enumerate(note_label_nums):
        if (entities[i][1], entities[i][0]) in processed_entities:
            if probs[i][note_label_num] > processed_entities[(entities[i][1], entities[i][0])]:
                used_indexes.append(i)  # reverse order
            else:
                used_indexes.append(i - 1)
        else:
            processed_entities[(entities[i][0], entities[i][1])] = probs[i][note_label_num]

    note_labels = [note_labels[x] for x in used_indexes]
    entities = [entities[x] for x in used_indexes]
    return event_timex_labels, note_labels, entities, offsets, tokens

if __name__ == '__main__':
    main()
