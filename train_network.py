'''
Training interface for Neural network model to detect and classify TLINKS between entities.
'''

import sys
import os
from src.config import env_paths
import numpy
numpy.random.seed(1337)

# this needs to be set. exit now so user doesn't wait to know.
if env_paths()["PY4J_DIR_PATH"] is None:
    sys.exit("PY4J_DIR_PATH environment variable not specified")

import argparse
import glob
import cPickle
import json
import threading
import Queue
import time

from src.learning.network import Network
from src.notes.TimeNote import TimeNote
from src.learning.word2vec import load_word2vec_binary, load_glove

from keras.models import model_from_json
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD


N_CLASSES = 13
EMBEDDING_DIM = 300
from src.learning.network import DENSE_LABELS

def main():
    '''
    Process command line arguments and then generate trained models (One for detection of links, one for classification)
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("train_dir",
                        help="Directory containing training annotations")

    parser.add_argument("model_destination",
                        help="Where to store the trained model")

    parser.add_argument("newsreader_annotations",
                        help="Where newsreader pipeline parsed file objects go")

    parser.add_argument("--val_dir",
                        default=None,
                        help="Directory containing validation annotations")

    parser.add_argument("--two_pass",
                        action='store_true',
                        default=False,
                        help="Train a single pass model that performs both detection and classification")

    parser.add_argument("--load_model",
                        action='store_true',
                        default=False,
                        help="Load saved model and resume training from there")

    parser.add_argument("--no_val",
                        action='store_true',
                        default=False,
                        help="No validation. Use all training data to train.")

    parser.add_argument("--pair_type",
                        default='both',
                        help="specify the entity type to train: intra, cross or both")

    parser.add_argument("--nolink",
                        default=None,
                        type=float,
                        help="no link downsampling ratio. e.g. 0.5 means # of nolinks are 50% of # positive tlinks")

    args = parser.parse_args()

    assert args.pair_type in ('intra', 'cross', 'both', 'dct')

    # validate file paths
    if os.path.isdir(args.newsreader_annotations) is False:
        sys.exit("invalid path for time note dir")
    if os.path.isdir(args.train_dir) is False:
        sys.exit("invalid path to directory containing training data")
    if os.path.isdir(os.path.dirname(args.model_destination)) is False:
        sys.exit("directory for model destination does not exist")

    print "arguments:\n"
    print args

    # get files in directory
    files = glob.glob(os.path.join(args.train_dir, '*'))
    gold_files = []
    tml_files  = []

    for f in files:
        if "E3input" in f:
            tml_files.append(f)
        elif f.endswith('.tml'):
            gold_files.append(f)

    gold_files.sort()
    tml_files.sort()

    if args.val_dir is None:
        val_files = None
    else:
        val_files = glob.glob(os.path.join(args.val_dir, '*'))
        val_files.sort()

    # one-to-one pairing of annotated file and un-annotated
    # assert len(gold_files) == len(tml_files)

    model_destination = os.path.join(args.model_destination, args.pair_type) + '/'
    if not os.path.exists(model_destination):
        os.makedirs(model_destination)

    if args.no_val:
        earlystopping = EarlyStopping(monitor='loss', patience=50, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint(model_destination + 'model.h5', monitor='loss', save_best_only=True)
    else:
        earlystopping = EarlyStopping(monitor='val_acc', patience=20, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint(model_destination + 'model.h5', monitor='val_loss', save_best_only=True)

    # create a sinlge model, then save architecture and weights
    if not args.two_pass:
        if args.load_model:
            try:
                NNet = load_model(model_destination + 'model.h5')
            except:
                NNet = model_from_json(open(model_destination + '.arch.json').read())
                #opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                #           decay=0.0)  # learning rate 0.001 is the default value
                opt = SGD(lr=0.003, momentum=0.9, decay=0.0, nesterov=False)
                NNet.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
                NNet.load_weights(model_destination + '.weights.h5')
        else:
            NNet = None

        NN, history = trainNetwork(gold_files, val_files, args.newsreader_annotations, args.pair_type,
                                   no_val=args.no_val, nolink_ratio=args.nolink,
                                   callbacks=[checkpoint, earlystopping], train_dir=args.train_dir)
        architecture = NN.to_json()
        open(model_destination + '.arch.json', "wb").write(architecture)
        NN.save_weights(model_destination + '.weights.h5')
        NN.save(model_destination + 'final_model.h5')
        json.dump(history, open(model_destination + 'training_history.json', 'w'))

    # # create a pair of models, one for detection, one for classification. Then save architecture and weights
    # else:
    #     # train models
    #     detector, classifier = trainNetwork(tml_files, gold_files, newsreader_dir)
    #
    #     # save models
    #     detect_arch = detector.to_json()
    #     class_arch = classifier.to_json()
    #     open(args.model_destination + '.detect.arch.json', "wb").write(detect_arch)
    #     open(args.model_destination + '.class.arch.json', "wb").write(class_arch)
    #     detector.save_weights(args.model_destination + '.detect.weights.h5')
    #     classifier.save_weights(args.model_destination + '.class.weights.h5')


def basename(name):
    name = os.path.basename(name)
    name = name.replace('.TE3input', '')
    name = name.replace('.tml', '')
    return name


def get_notes(files, newsreader_dir):

    if not files:
        return None

    notes = []
    if DENSE_LABELS:
        denselabels = cPickle.load(open(newsreader_dir+'dense-labels.pkl'))
    else:
        denselabels = None

    for i, tml in enumerate(files):
        if i % 10 == 0:
            print 'processing file {}/{} {}'.format(i + 1, len(files), tml)
        if os.path.isfile(os.path.join(newsreader_dir, basename(tml) + ".parsed.pickle")):
            tmp_note = cPickle.load(open(os.path.join(newsreader_dir, basename(tml) + ".parsed.pickle"), "rb"))
        else:
            tmp_note = TimeNote(tml, tml, denselabels=denselabels)
            cPickle.dump(tmp_note, open(newsreader_dir + "/" + basename(tml) + ".parsed.pickle", "wb"))

        if DENSE_LABELS and tmp_note.denselabels is None: # handle old note files without dense labels
            tmp_note.denselabels = denselabels
            tmp_note.get_id_to_denselabels()
        notes.append(tmp_note)
    return notes


class dataThread (threading.Thread):
    def __init__(self, threadID, inq, outq, word_vectors, pair_type, nolink_ratio, is_testdata=False):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.inq = inq
        self.outq = outq
        self.word_vectors = word_vectors
        self.pair_type = pair_type
        self.nolink_ratio = nolink_ratio
        self.is_testdata = is_testdata

    def run(self):
        print "Starting thread %d" %self.threadID
        network = Network()
        network.word_vectors = self.word_vectors
        while not self.inq.empty():
            notes = self.inq.get()
            if self.is_testdata:
                data = network._get_test_input(notes, pair_type=self.pair_type)
            else:
                data = network._get_training_input(notes, pair_type=self.pair_type, nolink_ratio=self.nolink_ratio, shuffle=True)
            self.outq.put(data)
            # time.sleep(0.5)
        print "Stopping thread %d" %self.threadID


def enqueue_notes(q, notes):
    n = 0
    N_notes = len(notes)
    while n < N_notes:
        if len(notes) <= 5:
            q.put(notes)
            break
        q.put(notes[0:5])
        del notes[0:5]
        n += 5


def dequeue_notes(q, is_testdata=False):
    Labels = []
    if is_testdata:
        counter = 0
        index_offset = 0
        while not q.empty():
            data = q.get()
            if not Labels:
                XL, XR, Labels, Pair_Index = data
            else:
                xl, xr, labels, pair_index = data
                XL = Network._pad_and_concatenate(XL, xl, axis=0, pad_left=[1])
                XR = Network._pad_and_concatenate(XR, xr, axis=0, pad_left=[1])
                Labels += labels
                for key, value in pair_index.items():
                    note_id, pair = key
                    Pair_Index[(note_id + 5*counter, pair)] = value + index_offset
            counter += 1
            index_offset = len(Pair_Index)
        return XL, XR, Labels, Pair_Index
    else:
        while not q.empty():
            data = q.get()
            if not Labels:
                XL, XR, Labels = data
            else:
                xl, xr, labels = data
                XL = Network._pad_and_concatenate(XL, xl, axis=0, pad_left=[1])
                XR = Network._pad_and_concatenate(XR, xr, axis=0, pad_left=[1])
                Labels += labels
        return XL, XR, Labels


def trainNetwork(gold_files, val_files, newsreader_dir, pair_type, no_val=False, nolink_ratio=1.0, callbacks=[], train_dir='./'):
    '''
    train::trainNetwork()
    Purpose: Train a neural network for classification of temporal realtions. Assumes events and timexes
        will be provided at prediction time
    @param tml_files: List of unlabled (no timex, etc) timeML documents
    @param gold_files: Fully labeled gold standard timeML documents
    '''

    print "Called trainNetwork"

    global N_CLASSES

    if not os.path.isfile(train_dir+'training_data.pkl'):
        notes = get_notes(gold_files, newsreader_dir)
    if not no_val:
        val_notes = get_notes(val_files, newsreader_dir)

    print "loading word vectors..."
    word_vectors = load_word2vec_binary(os.environ["TEA_PATH"] + '/GoogleNews-vectors-negative300.bin', verbose=0)
    # network.word_vectors = load_glove(os.environ["TEA_PATH"] + '/glove.6B.200d.txt')
    # network.word_vectors = load_glove(os.environ["TEA_PATH"] + '/glove.6B.300d.txt')
    network = Network()
    network.word_vectors = word_vectors

    inq = Queue.Queue()
    outq = Queue.Queue()
    enqueue_notes(inq, notes)

    threads = []
    n_threads = min(4, inq.qsize())
    for t in range(n_threads):
        data_thread = dataThread(t, inq, outq, word_vectors, pair_type, nolink_ratio)
        data_thread.start()
        threads.append(data_thread)

    while not inq.empty():
        n_notes = inq.qsize() * 5
        sys.stdout.write("# notes in queue: %d \r" %n_notes)
        sys.stdout.flush()
        time.sleep(1)

    for t in threads:
        t.join()

    training_data = dequeue_notes(outq)

    print "training data size:", training_data[0].shape
    print "training data labels", training_data[2][:100]

    if not no_val and val_notes is not None:
        val_data = network._get_test_input(val_notes, pair_type=pair_type)
        print "validation data size:", val_data[0].shape
    else:
        val_data = None

    #cPickle.dump(data, open(train_dir+'training_data.pkl', 'w'))

    del network.word_vectors
    if DENSE_LABELS:
        NNet, history = network.train_model(None, epochs=200, training_input=training_data, val_input=val_data, no_val=no_val, weight_classes=True, batch_size=32,
        encoder_dropout=0, decoder_dropout=0.5, input_dropout=0.6, reg_W=0, reg_B=0, reg_act=0, LSTM_size=128,
        dense_size=100, maxpooling=True, data_dim=EMBEDDING_DIM, max_len='auto', nb_classes=N_CLASSES, callbacks=callbacks)
    else:
        NNet, history = network.train_model(None, epochs=200, training_input=training_data, val_input=val_data, no_val=no_val, weight_classes=False, batch_size=64,
        encoder_dropout=0, decoder_dropout=0.5, input_dropout=0.6, reg_W=0, reg_B=0, reg_act=0, LSTM_size=256,
        dense_size=100, maxpooling=True, data_dim=EMBEDDING_DIM, max_len='auto', nb_classes=N_CLASSES, callbacks=callbacks)

    return NNet, history

if __name__ == "__main__":
  main()
