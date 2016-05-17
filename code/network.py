import os
import pickle

import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Graph
from keras.layers import Embedding, LSTM, Dense, Merge, MaxPooling1D, TimeDistributedDense, Flatten, Masking, Input, Permute
#from notes.TimeNote import TimeNote
from gensim.models import word2vec

class NNModel:

    def __init__(self, data_dim=300, max_len=22, nb_classes=7):
        '''
        Creates a neural network with the specified conditions.
        '''
        # encode the first entity
        encoder_L = Sequential()
        # encoder_L.add(Masking(mask_value=0., input_shape=(data_dim, max_len)))
        encoder_L.add(LSTM(300, input_shape=(data_dim, max_len), return_sequences=True))
        encoder_L.add(MaxPooling1D(pool_length=300))
        encoder_L.add(Flatten())

        # encode the second entity
        encoder_R = Sequential()
        # encoder_R.add(Masking(mask_value=0., input_shape=(data_dim, max_len)))
        encoder_R.add(LSTM(300, input_shape=(data_dim, max_len), return_sequences=True))
        encoder_R.add(MaxPooling1D(pool_length=300))
        encoder_R.add(Flatten())

        # combine and classify entities as a single relation
        decoder = Sequential()
        decoder.add(Merge([encoder_R, encoder_L], mode='concat'))
        decoder.add(Dense(100, activation='relu'))
        decoder.add(Dense(nb_classes, activation='softmax'))

        # compile the final model
        decoder.compile(loss='categorical_crossentropy', optimizer='adam')
        self.classifier = decoder

    def train(self, notes, epochs=5):
        '''
        obtains entity pairs and tlink labels from every note passed, and uses them to train the network.
        '''

        # TODO: handle tlinks linking to the document creation time. at the moment, we simply skip them

        print "\ninput_shape: ", self.classifier.input_shape, "\n"

        # labels for each SDP pair
        tlinklabels = []

        # data tensor for left and right SDP subpaths
        XL = None
        XR = None

        print 'Loading word embeddings...'
        word_vectors = word2vec.Word2Vec.load_word2vec_format(os.environ["TEA_PATH"]+'/GoogleNews-vectors-negative300.bin', binary=True)

        print 'Extracting dependency paths...'
        for i, note in enumerate(notes):
            # get tlink lables
            note_tlinklabels = note.get_tlink_labels()

            # get the representation for the event/timex pairs in the note
            # will be 3D tensor with axis zero holding the each pair, axis 1 holding the word embeddings
            # (with length equal to word embedding length), and axis 2 hold each word.
            # del_list is a list of indices for which no SDP could be obtained
            left_vecs, right_vecs, del_list = _extract_path_representations(note, word_vectors)

            # add the note's data to the combine data matrix
            if XL == None:
                XL = left_vecs
            else:
                XL = _pad_and_concatenate(XL, left_vecs, axis=0)

            if XR == None:
                XR = right_vecs
            else:
                XR = _pad_and_concatenate(XR, right_vecs, axis=0)

            # remove duplicate indices
            del_list = list(set(del_list))
            # remove indices in descending order so that they continue to refer to the item we want to remove
            del_list.sort()
            del_list.reverse()
            for index in del_list:
                del note_tlinklabels[index]

            # add remaining labels to complete list of labels
            tlinklabels += note_tlinklabels

        # reformat labels so that they can be used by the NN
        labels = _convert_str_labels_to_int(tlinklabels)
        Y = to_categorical(labels,7)

        # pad XL and XR so that they have the same number of dimensions on the second axis
        # any other dimension mis-matches are caused by actually errors and should not be padded away
        XL, XR = _pad_to_match_dimensions(XL, XR, 2)

        class_weights = _get_uniform_weights(Y)
        print class_weights

        # train the network
        print 'Training network...'
        self.classifier.fit([XL, XR], Y, nb_epoch=epochs, validation_split=0.15, class_weight=class_weights, batch_size=256)

        test = self.classifier.predict_classes([XL, XR])

        print test

        T = 0
        F = 0
        outs = 0
        for true, pred in zip(labels, test):
            if true == pred:
                T += 1
            else:
                F += 1
            if true == 0:
                outs += 1

        print "T: ", T, "F: ", F, "outs: ", outs

    def predict(self, notes):
        '''
        use the trained model to predict the labels of some data
        '''

        # data tensors for left and right SDP
        XL = None
        XR = None

        # list of lists, where each list contains the indices to delete from a given note
        # indices of the outer list corespond to indices of the notes list
        del_lists = []

        print 'Loading word embeddings...'
        word_vectors = word2vec.Word2Vec.load_word2vec_format(os.environ["TEA_PATH"]+'/GoogleNews-vectors-negative300.bin', binary=True)

        print 'Extracting dependency paths...'
        for i, note in enumerate(notes):
            # get the representation for the event/timex pairs in the note
            # will be 3D tensor with axis zero holding the each pair, axis 1 holding the word embeddings
            # (with length equal to word embedding length), and axis 2 hold each word.
            left_vecs, right_vecs, del_list = _extract_path_representations(note, word_vectors)

            # add the list of indices to delete from every file to the primary list
            del_lists.append(del_list)

            # add the note's data to the combine data matrix
            if XL == None:
                XL = left_vecs
            else:
                XL = _pad_and_concatenate(XL, left_vecs, axis=0)

            if XR == None:
                XR = right_vecs
            else:
                XR = _pad_and_concatenate(XR, right_vecs, axis=0)

        # pad XL and XR so that they have the same number of dimensions on the second axis
        # any other dimension mis-matches are caused by actually errors and should not be padded away
        XL, XR = _pad_to_match_dimensions(XL, XR, 2)

        filler = np.ones((22,22,22))

        XL, _ = _pad_to_match_dimensions(XL, filler, 2)
        XR, _ = _pad_to_match_dimensions(XR, filler, 2)

        print 'Predicting...'
        labels = self.classifier.predict_classes([XL, XR])

        return _convert_int_labels_to_str(labels), del_lists

def _extract_path_representations(note, word_vectors):
    '''
    convert a note into a portion of the input matrix
    '''

    # retrieve the ids for the left and right halves of every SDP between tlinked entities
    left_ids, right_ids = _get_token_id_subpaths(note)

    # left and right paths are used to store the actual tokens of the SDP paths
    left_paths = []
    right_paths = []

    # del list stores the indices of pairs which do not have a SDP so that they can be removed from the labels later
    del_list = []

    # get token text from ids in left sdp
    for id_list in left_ids:
        left_paths.append(note.get_tokens_from_ids(id_list))

    # get token text from ids in right sdp
    for id_list in right_ids:
        right_paths.append(note.get_tokens_from_ids(id_list))

    # get the word vectors for every word in the left path
    left_vecs = None
    for j, path in enumerate(left_paths):
        vecs_path = None
        for word in path:
            # try to get embedding for a given word. If the word is not in the vocabulary, use a vector of all 1s.
            try:
                embedding = np.asarray(word_vectors[word], dtype='float32')
            except KeyError:
                embedding = np.ones((300))

            # reshape to 3 dimensions so embeddings can be concatenated together to form the final input values
            embedding = embedding[np.newaxis, :, np.newaxis]
            if vecs_path == None:
                vecs_path = embedding
            else:
                vecs_path = np.concatenate((vecs_path, embedding), axis=2)

        # if there were no vectors, the link involves the document creation time or is a cross sentence relation.
        # add index to list to indexes to remove and continue
        if vecs_path == None:
            del_list.append(j)
            continue
        if left_vecs == None:
            left_vecs = vecs_path
        else:
            left_vecs = _pad_and_concatenate(left_vecs, vecs_path, axis=0)

    # get the vectors for every word in the right path
    right_vecs = None
    for j, path in enumerate(right_paths):
        vecs_path = None
        for word in path:
            # try to get embedding for a given word. If the word is not in the vocabulary, use a vector of all 0s.
            try:
                embedding = np.asarray(word_vectors[word], dtype='float32')
            except KeyError:
                embedding = np.zeros((300))

            # reshape to 3 dimensions so embeddings can be concatenated together to form the final input values
            embedding = embedding[np.newaxis, :, np.newaxis]
            if vecs_path == None:
                vecs_path = embedding
            else:
                vecs_path = np.concatenate((vecs_path, embedding), axis=2)

        # if there were no vectors, the link involves the document creation time or is a cross sentence relation.
        # remove label from list and continue to the next path
        if vecs_path == None:
            del_list.append(j)
            continue
        if right_vecs == None:
            right_vecs = vecs_path
        else:
            right_vecs = _pad_and_concatenate(right_vecs, vecs_path, axis=0)

    return left_vecs, right_vecs, del_list

def _pad_and_concatenate(a, b, axis):
    '''
    given two tensors, pad with zeros so that all dimensions are equal except the axis to concatentate on, and the concatenate
    '''
    # if tensors have number of dimensions
    if len(a.shape) != len(b.shape):
        pass

    # for each axis that is not to be concatenated on, pad the smaller of the two tensors with zeros so that both have equal sizes
    for i in range(len(a.shape)):
        if i == axis:
            continue
        a, b = _pad_to_match_dimensions(a, b, i)
    # concatenate and return the padded matrices
    return np.concatenate((a, b), axis=axis)

def _pad_to_match_dimensions(a, b, axis):
    '''
    given to tensors and an axis for comparison, pad the smaller of the two with zeros to match the dimensions of the larger
    '''

    a_axis = a.shape[axis]
    b_axis = b.shape[axis]
    if a_axis > b_axis:
        pad_shape = list(b.shape)
        pad_shape[axis] = a_axis - b_axis
        b = np.concatenate((b, np.zeros(tuple(pad_shape))), axis=axis)
    if b_axis > a_axis:
        pad_shape = list(a.shape)
        pad_shape[axis] = b_axis - a_axis
        a = np.concatenate((a, np.zeros(tuple(pad_shape))), axis=axis)

    return a, b


def _get_token_id_subpaths(note):
    '''
    extract ids for the tokens in each half of the shortest dependency path between each token in each relation
    '''
    # TODO: for now we only look at the first token in a given entity. Eventually, we should get all tokens in the entity

    pairs = note.get_tlinked_entities()

    left_paths = []
    right_paths = []

    for pair in pairs:
        target_id = ''
        source_id = ''

        # extract and reformat the ids in the pair to be of form t# instead of w#
        # events may be linked to document creation time, which will not have an id
        if 'id' in pair["target_entity"][0]:
            target_id = pair["target_entity"][0]["id"]
            target_id = 't' + target_id[1:]
        if 'id' in pair["src_entity"][0]:
            source_id = pair["src_entity"][0]["id"]
            source_id = 't' + source_id[1:]

        left_path, right_path = note.dependency_paths.get_left_right_subpaths(target_id, source_id)
        left_paths.append(left_path)
        right_paths.append(right_path)

    return left_paths, right_paths

def _get_uniform_weights(labels):
    '''
    get a dictionary of weights for each class. Used to combat imbalanced data problems by reducing
    the impact of highly represented classes. Has no effect on equally distributed data
    '''
    # Y is composed of one-hot vectors for each class
    classes = {}
    for label in labels:
        for i, _class in enumerate(label):
            if _class == 1:
                if i in classes:
                    classes[i] += 1.0
                else:
                    classes[i] = 1.0

    for _class in classes:
        classes[_class] /= len(labels)
        classes[_class] = 1 - classes[_class]

    return classes

def _convert_str_labels_to_int(labels):
    '''
    convert tlink labels to integers so they can be processed by the network
    '''

    processed_labels = []
    for label in labels:
        if label == "SIMULTANEOUS":
            processed_labels.append(1)
        elif label == "BEFORE":
            processed_labels.append(2)
        elif label == "AFTER":
            processed_labels.append(3)
        elif label == "IS_INCLUDED":
            processed_labels.append(4)
        elif label == "BEGUN_BY":
            processed_labels.append(5)
        elif label == "ENDED_BY":
            processed_labels.append(6)
        else:  # label for pairs which are not linked
            processed_labels.append(0)

    return processed_labels

def _convert_int_labels_to_str(labels):
    '''
    convert ints to tlink labels so network output can be understood
    '''

    processed_labels = []
    for label in labels:
        if label == 1:
            processed_labels.append("SIMULTANEOUS")
        elif label == 2:
            processed_labels.append("BEFORE")
        elif label == 3:
            processed_labels.append("AFTER")
        elif label == 4:
            processed_labels.append("IS_INCLUDED")
        elif label == 5:
            processed_labels.append("BEGUN_BY")
        elif label == 6:
            processed_labels.append("ENDED_BY")
        else:  # label for unlinked pairs (should have int 0)
            processed_labels.append("None")

    return processed_labels

if __name__ == "__main__":
    test = NNModel()
    with open("note.dump") as n:
        tmp_note = pickle.load(n)
#    tmp_note = TimeNote("APW19980418.0210.tml.TE3input", "APW19980418.0210.tml")
    # print tmp_note.pre_processed_text[2][16]
#    with open("note.dump", 'wb') as n:
#        pickle.dump(tmp_note, n)
    test.train([tmp_note])

    # labels = tmp_note.get_tlink_labels()
    # labels = _convert_str_labels_to_int(labels)
    # _labels = to_categorical(labels,7)
    # print len(labels)
    # print labels
    # input1 = np.random.random((len(labels),300, 16))
    # input2 = np.random.random((len(labels),300, 16))
    # # labels = np.random.randint(7, size=(10000,1))
    # test.classifier.fit([input1,input2], _labels, nb_epoch=100)
    # print test.classifier.predict_classes([input1,input2])
    # print labels
    pass
