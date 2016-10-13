'''
Training interface for Neural network model to detect and classify TLINKS between entities.
'''

import sys
import os
from code.config import env_paths

# this needs to be set. exit now so user doesn't wait to know.
if env_paths()["PY4J_DIR_PATH"] is None:
    sys.exit("PY4J_DIR_PATH environment variable not specified")

import argparse
import glob
import cPickle

from code.learning import network_sem10

entnote_imported = False

def main():
    '''
    Process command line arguments and then generate trained models (One for detection of links, one for classification)
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("training_file",
                        type=str,
                        nargs=1,
                        help="Directory of training file")

    parser.add_argument("model_destination",
                        help="Where to store the trained model")

    parser.add_argument("newsreader_annotations",
                        help="Where newsreader pipeline parsed file objects go")

    parser.add_argument("--single_pass",
                        action='store_true',
                        default=False,
                        help="Train a single pass model that performs both detection and classification")

    args = parser.parse_args()

    newsreader_dir = args.newsreader_annotations

    print "training file:", args.training_file[0]
    print "newreader_dir", newsreader_dir
    # validate file paths
    if os.path.isfile(args.training_file[0]) is False:
        sys.exit("training file for semeval 10 task 8 not found")
    if os.path.isdir(os.path.dirname(args.model_destination)) is False:
        sys.exit("directory for model destination does not exist")

    gold_file = args.training_file[0]

    # create a sinlge model, then save architecture and weights
    if args.single_pass:
        NN = trainNetwork(gold_file, newsreader_dir, two_pass=False)
        architecture = NN.to_json()
        open(args.model_destination + '.arch.json', "wb").write(architecture)
        NN.save_weights(args.model_destination + '.weights.h5')

    # create a pair of models, one for detection, one for classification. Then save architecture and weights
    else:
        # train models
        detector, classifier = trainNetwork(gold_file, newsreader_dir)

        # save models
        detect_arch = detector.to_json()
        class_arch = classifier.to_json()
        open(args.model_destination + '.detect.arch.json', "wb").write(detect_arch)
        open(args.model_destination + '.class.arch.json', "wb").write(class_arch)
        detector.save_weights(args.model_destination + '.detect.weights.h5')
        classifier.save_weights(args.model_destination + '.class.weights.h5')


def trainNetwork(gold_file, newsreader_dir, two_pass=True):
    '''
    train::trainNetwork()

    Purpose: Train a neural network for classification of realtions. Assumes events and timexes
        will be provided at prediction time

    @param gold_file: training file containing sentences and relations
    '''

    print "Called trainNetwork"

    global entnote_imported

    basename = os.path.splitext(gold_file)[0].split('/')[-1] # filename without directory and extension
    # Read in notes
    notes = []

    note_file = os.path.join(newsreader_dir, basename+".parsed.pickle")
    if os.path.isfile(note_file):
        ent_note = cPickle.load(open(note_file, "rb"))
    else:
        if entnote_imported is False:
            from code.notes.EntNote import EntNote
            entnote_imported = True
        ent_note = EntNote(gold_file, overwrite=False)
        cPickle.dump(ent_note, open(note_file, "wb"))

    notes.append(ent_note) # use a list here because the interface can read multiple notes

    if two_pass:

        detect_data = network_sem10._get_training_input(notes, presence=True, no_none=False)
        classify_data = network_sem10._get_training_input(notes, presence=False, no_none=True)

        detector = network_sem10.train_model(None, epochs=150, training_input=detect_data, weight_classes=False, batch_size=256,
        encoder_dropout=0, decoder_dropout=0, input_dropout=0.5, reg_W=0, reg_B=0, reg_act=0, LSTM_size=64, dense_size=100, maxpooling=True, data_dim=300, max_len='auto', nb_classes=2)

        # use max input length from detector
        max_len = detector.input_shape[0][2]

        classifier = network_sem10.train_model(None, epochs=500, training_input=classify_data, weight_classes=False, batch_size=256,
        encoder_dropout=0., decoder_dropout=0., input_dropout=0.5, reg_W=0, reg_B=0, reg_act=0, LSTM_size=64, dense_size=100, maxpooling=True, data_dim=300, max_len=max_len, nb_classes=19)

        return detector, classifier

    else:

        data = network_sem10._get_training_input(notes)

        NNet = network_sem10.train_model(None, epochs=150, training_input=data, weight_classes=False, batch_size=256,
        encoder_dropout=0, decoder_dropout=0, input_dropout=0.5, reg_W=0, reg_B=0, reg_act=0, LSTM_size=64, dense_size=100, maxpooling=True, data_dim=300, max_len='auto', nb_classes=20)

        return NNet

if __name__ == "__main__":
  main()
