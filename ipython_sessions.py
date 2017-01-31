# coding: utf-8
#%load_ext autoreload
#%autoreload 2

import os
from code.config import env_paths
import cPickle
from code.notes import TimeNote
#gold='ABC19980114.1830.0611.tml'
#tml='ABC19980114.1830.0611.tml.TE3input'
#note = TimeNote.TimeNote('training_data_augmented/ABC19980304.1830.1636.tml.TE3input', 'training_data_augmented/ABC19980304.1830.1636.tml')
note=cPickle.load(open('newsreader_annotations/1-20/AP_20130322.parsed.pickle'))
#note1=cPickle.load(open('newsreader_annotations/1-20/ABC19980108.1830.0711.parsed.pickle'))
#note2=cPickle.load(open('newsreader_annotations/1-20/APW19980213.1380.parsed.pickle'))
#model_path='model_destination/tea/12-23/5classes/'
#from keras.models import model_from_json
#NNet = model_from_json(open(model_path + '.arch.json').read())
#NNet.load_weights(model_path + '.weights.h5')
#notes=[note, note1, note2]
#from code.learning.network import Network
#network = Network()
#labels, filter_lists, probs = network.single_predict(notes, NNet, evalu=False, predict_prob=True)
#from code.learning.word2vec import load_word2vec_binary
#word_vectors = load_word2vec_binary(os.environ["TEA_PATH"]+'/GoogleNews-vectors-negative300.bin', verbose=0)
#XL,YL,labels=network._get_training_input(notes, shuffle=False)

#from code.notes.utilities import timeml_utilities
#from code.learning.model_event import EventWriter
#writer=EventWriter(note)
#tagged=writer.tag_predicates()

#from code.notes.utilities.xml_utilities import write_root_to_file
from code.learning.time_ref import TimeRefNetwork
from code.learning.time_ref import predict_timex_rel
#tr=TimeRefNetwork(note)
#tr1=TimeRefNetwork(note1)
#tr2=TimeRefNetwork(note2)

#predictions = predict_timex_rel(notes)
#print "predicting done."

