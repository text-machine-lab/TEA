"""
Interface to perform predicting of TIMEX, EVENT and TLINK annotations,
using a neural network model for TLINK annotations.
"""

import sys
from code.config import env_paths
import numpy
# numpy.random.seed(1337)

# this needs to be set. exit now so user doesn't wait to know.
if env_paths()["PY4J_DIR_PATH"] is None:
    sys.exit("PY4J_DIR_PATH environment variable not specified")

import argparse
import cPickle
import glob
import os
import Queue
import threading

from keras.models import model_from_json, load_model
from sklearn.metrics import classification_report

from code.learning.network import Network
from code.notes.TimeNote import TimeNote
from code.learning.time_ref import predict_timex_rel
from code.learning.break_cycle import modify_tlinks
from code.learning.word2vec import load_word2vec_binary, load_glove
from train_network import dataThread
from train_networkmem import get_notes, MAX_LEN

# timenote_imported = False
from code.learning.network_mem import NetworkMem, DENSE_LABELS, LABELS


def basename(s):
    if '.tml' in s:
        s = os.path.basename(s[0:s.index(".tml")])
    if 'TE3input' in s:
        s = os.path.basename(s[0:s.index(".TE3input")])
    return s


def main():

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

    parser.add_argument("--no_ntm",
                        action='store_true',
                        default=False,
                        help="specify whether to use neural turing machine. default is to use ntm (no_ntm=false).")

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

    # intra_model = model_from_json(open(os.path.join(args.intra_model_path, 'intra', '.arch.json')).read())
    # intra_model.load_weights(os.path.join(args.intra_model_path, 'intra', '.weights.h5'))
    intra_model = load_model(os.path.join(args.intra_model_path, 'intra', 'model.h5'))
    # cross_model = model_from_json(open(os.path.join(args.cross_model_path, 'cross', '.arch.json')).read())
    # cross_model.load_weights(os.path.join(args.cross_model_path, 'cross', '.weights.h5'))
    cross_model = load_model(os.path.join(args.cross_model_path, 'cross', 'model.h5'))
    # dct_model = model_from_json(open(os.path.join(args.dct_model_path, 'dct', '.arch.json')).read())
    # dct_model.load_weights(os.path.join(args.dct_model_path, 'dct', '.weights.h5'))
    dct_model = load_model(os.path.join(args.dct_model_path, 'dct', 'model.h5'))

    notes = get_notes(gold_files, newsreader_dir)

    network = NetworkMem()
    network.word_vectors = load_word2vec_binary(os.environ["TEA_PATH"] + 'embeddings/GoogleNews-vectors-negative300.bin', verbose=0)

    if DENSE_LABELS:
        pred_Y = []
        true_Y = []
        denselabels = cPickle.load(open(newsreader_dir + 'dense-labels.pkl'))
    else:
        denselabels = None

    for note in notes:
        # predict only one note each time, otherwise the indexes need to be changed
        pred_y, true_y = predict_note([note], network, intra_model, cross_model, dct_model, annotation_destination,
                     denselabels=denselabels, no_ntm=args.no_ntm)
        if DENSE_LABELS:
            pred_Y += pred_y
            true_Y += true_y

    print "All predictions written to %s" %annotation_destination
    if DENSE_LABELS:
        network.class_confusion(pred_Y, true_Y, len(LABELS))
        print "my calculation:"
        true_Y = numpy.array(true_Y)
        pred_Y = numpy.array(pred_Y)
        diff = numpy.count_nonzero(true_Y - pred_Y)
        print "ACC = ", (len(pred_Y)-diff)*1.0/ len(pred_Y)



def predict_note(notes, network, intra_model, cross_model, dct_model, annotation_destination, denselabels=None, no_ntm=False):


    timex_labels, timex_pair_index = predict_timex_rel(notes)

    data_gen = network.generate_test_input(notes, 'intra', max_len=MAX_LEN, no_ntm=no_ntm)
    intra_labels, intra_probs, intra_pair_index = network.predict(intra_model, data_gen, batch_size=300,
                                                                  evaluation=False, smart=True, no_ntm=no_ntm)

    data_gen = network.generate_test_input(notes, 'cross', max_len=MAX_LEN, no_ntm=no_ntm)
    cross_labels, cross_probs, cross_pair_index = network.predict(cross_model, data_gen, batch_size=500,
                                                                  evaluation=False, smart=True, no_ntm=no_ntm)

    data_gen = network.generate_test_input(notes, 'dct', max_len=MAX_LEN, no_ntm=no_ntm)
    dct_labels, dct_probs, dct_pair_index = network.predict(dct_model, data_gen, batch_size=50,
                                                                  evaluation=False, smart=True, no_ntm=no_ntm)
    intra_labels = network._convert_int_labels_to_str(intra_labels)
    print intra_labels[:10]
    intra_scores = [max(probs) for probs in intra_probs]
    cross_labels = network._convert_int_labels_to_str(cross_labels)
    cross_scores = [max(probs) for probs in cross_probs]
    dct_labels = network._convert_int_labels_to_str(dct_labels)
    dct_scores = [max(probs) for probs in dct_probs]

    assert len(dct_labels) == len(dct_scores)

    for i, note in enumerate(notes):
        note_id_pairs = []
        note_labels = []
        note_scores = []

        for key in timex_pair_index.keys():  # {(note_id, (t, t)) : index}
            if key[0] == i:
                note_id_pairs.append(key[1])
                note_labels.append(timex_labels[timex_pair_index[key]])
                note_scores.append(1.0)  # trust timex tlinks
                timex_pair_index.pop(key)

        for key in dct_pair_index.keys():  # {(note_id, (ei, t0)) : index}
            if key[0] == i:
                note_id_pairs.append(key[1])
                note_labels.append(dct_labels[dct_pair_index[key]])
                note_scores.append(max(dct_probs[dct_pair_index[key]]))
                # note_scores.append(0.0)
                dct_pair_index.pop(key)
        # print "modifying DCT tlinks..."
        # note_labels = modify_tlinks(note_id_pairs, note_labels, note_scores)

        for key in intra_pair_index.keys():  # {(note_id, (ei, ej)) : index}
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
        # print "modifying intra-sentence tlinks..."
        # note_labels = modify_tlinks(note_id_pairs, note_labels, note_scores)

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

        # note_labels, note_scores = resolve_coref(note, note_id_pairs, note_labels, note_scores)
        print "modifying final tlinks..."
        note_labels = modify_tlinks(note_id_pairs, note_labels, note_scores)
        save_predictions(note, note_id_pairs, note_labels, annotation_destination)

        pred_y = []
        true_y = []
        if denselabels is not None:
            filename = basename(note.annotated_note_path)
            note_denselabels = denselabels[filename]
            for id_pair, label in zip(note_id_pairs, network._convert_str_labels_to_int(note_labels)):
                if id_pair in note_denselabels:
                    pred_y.append(label)
                    true_y += network._convert_str_labels_to_int([note_denselabels[id_pair]])
                elif (id_pair[1], id_pair[0]) in note_denselabels:
                    pred_y += network.reverse_labels([label])
                    true_y += network._convert_str_labels_to_int([note_denselabels[(id_pair[1], id_pair[0])]])
                else:
                    print "pair not found in dense labels:", id_pair, label

            # examination
            for pair in note_denselabels:
                if pair not in note_id_pairs and (pair[1], pair[0]) not in note_id_pairs:
                    print "dense pairs not found in predicted labels", pair, note_denselabels[pair]

        return pred_y, true_y

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
    print "\nsaving predictions in", note_path
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

if __name__ == '__main__':
    main()
