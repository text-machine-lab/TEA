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

from keras.models import model_from_json, load_model
from sklearn.metrics import classification_report

from code.notes.TimeNote import TimeNote
from code.learning.time_ref import predict_timex_rel
from code.learning.break_cycle import modify_tlinks
from train_networkmem import get_notes, MAX_LEN

# timenote_imported = False
from code.learning.network_mem import NetworkMem, DENSE_LABELS, LABELS
from code.learning.network import Network
HAS_AUX = False

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

    parser.add_argument("model_path",
                        help="Where trained model is located")

    parser.add_argument("annotation_destination",
                         help="Where annotated files are written")

    parser.add_argument("newsreader_annotations",
                        help="Where newsreader pipeline parsed file objects go")

    parser.add_argument("--no_ntm",
                        action='store_true',
                        default=False,
                        help="specify whether to use neural turing machine. default is to use ntm (no_ntm=false).")

    parser.add_argument("--eval",
                        action='store_true',
                        default=False,
                        help="specify whether to evaluate results against ground truth.")

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
    notes = get_notes(gold_files, newsreader_dir, augment=True)

    network = NetworkMem(no_ntm=args.no_ntm, nb_training_files=len(notes))
    network.word_vectors = cPickle.load(open(os.path.join(args.model_path, 'all/vocab.pkl')))
    try:
        model = load_model(os.path.join(args.model_path, 'all/model.h5'))
    except:
        model = network.load_raw_model(False)
        model.load_weights(os.path.join(args.model_path, 'all/final_weights.h5'))


    if DENSE_LABELS:
        denselabels = cPickle.load(open(newsreader_dir + 'dense-labels.pkl'))
        # denselabels = cPickle.load(open(newsreader_dir + 'dense-labels-single.pkl'))
    else:
        denselabels = None

    predict_note(notes, network, model, annotation_destination, denselabels=denselabels, no_ntm=args.no_ntm, eval=args.eval)

    # for note in notes:
    #     # predict only one note each time, otherwise the indexes need to be changed
    #     pred_y, true_y = predict_note([note], network, intra_model, cross_model, dct_model, annotation_destination,
    #                  denselabels=denselabels, no_ntm=args.no_ntm)
    #     if DENSE_LABELS:
    #         pred_Y += pred_y
    #         true_Y += true_y
    #
    # print "All predictions written to %s" %annotation_destination
    # if DENSE_LABELS:
    #     network.class_confusion(pred_Y, true_Y, len(LABELS))
    #     print "my calculation:"
    #     true_Y = numpy.array(true_Y)
    #     pred_Y = numpy.array(pred_Y)
    #     diff = numpy.count_nonzero(true_Y - pred_Y)
    #     print "ACC = ", (len(pred_Y)-diff)*1.0/ len(pred_Y)



def predict_note(notes, network, model, annotation_destination, denselabels=None, no_ntm=False, eval=False):

    test_data_gen = network.generate_test_input(notes, 'all', max_len=MAX_LEN, no_ntm=no_ntm, multiple=1)
    predictions, scores, true_labels, pair_indexes = network.predict(model, test_data_gen, batch_size=0,
                                                                     evaluation=False, smart=True, no_ntm=no_ntm, pruning=False)

    if denselabels is not None:
        # map the results to original pairs
        # After double-check, some pairs may have been flipped. we flip them back for here.
        note_denselabels = []
        for note_id in range(len(notes)):
            filename = basename(notes[note_id].annotated_note_path)
            note_denselabels.append(denselabels[filename])

        for k in pair_indexes:
            note_id, pair = k
            if pair not in note_denselabels[note_id]:
                if (pair[1], pair[0]) in note_denselabels[note_id]:
                    index = pair_indexes[(note_id, pair)]
                    pair_indexes[(note_id, (pair[1], pair[0]))] = index
                    predictions[index] = network.reverse_labels([predictions[index]])[0]
                    true_labels[index] = network.reverse_labels([true_labels[index]])[0]
                    pair_indexes.pop((note_id, pair))
                else:
                    print("pair not found in dense labels:", basename(notes[note_id].annotated_note_path), pair)

    if eval:
        Network.class_confusion(predictions, true_labels, len(LABELS))

    for i, note in enumerate(notes):
        note_id_pairs = [pair for note_id, pair in pair_indexes if note_id == i]
        note_labels = [predictions[pair_indexes[(i, pair)]] for pair in note_id_pairs]
        note_labels_str = network.convert_int_labels_to_str(note_labels)

        save_predictions(note, note_id_pairs, note_labels_str, annotation_destination)

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
            elif '<SLINK' in line:
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
