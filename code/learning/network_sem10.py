import os
import cPickle as pickle
import json
import copy

import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Graph
from keras.layers import Embedding, LSTM, Dense, Merge, MaxPooling1D, TimeDistributed, Flatten, Masking, Input, Dropout
from keras.regularizers import l2, activity_l2
from sklearn.metrics import classification_report

from word2vec import load_word2vec_binary

def set_ignore_order(no_order):
    global ignore_order
    ignore_order = no_order

def get_untrained_model(encoder_dropout=0, decoder_dropout=0, input_dropout=0, reg_W=0, reg_B=0, reg_act=0, LSTM_size=32, dense_size=100, maxpooling=True, data_dim=300, max_len=22, nb_classes=7):
    '''
    Creates a neural network with the specified conditions.
    params:
        encoder_dropout: dropout rate for LSTM encoders (NOT dropout for LSTM internal gates)
        decoder_dropout: dropout rate for decoder
        reg_W: lambda value for weight regularization
        reg_b: lambda value for bias regularization
        reg_act: lambda value for activation regularization
        LSTM_size: number of units in the LSTM layers
        maxpooling: pool over LSTM output at each timestep, or just take the output from the final LSTM timestep
        data_dim: dimension of the input data
        max_len: maximum length of an input sequence (this should be found based on the training data)
        nb_classes: number of classes present in the training data
    '''

    # create regularization objects if needed
    if reg_W != 0:
        W_reg = l2(reg_W)
    else:
        W_reg = None

    if reg_B != 0:
        B_reg = l2(reg_B)
    else:
        B_reg = None

    if reg_act != 0:
        act_reg = activity_l2(reg_act)
    else:
        act_reg = None

    # encode the first entity
    encoder_L = Sequential()

    encoder_L.add(Dropout(input_dropout, input_shape=(data_dim, max_len)))

    # with maxpooling
    if maxpooling:
        encoder_L.add(LSTM(LSTM_size, return_sequences=True, inner_activation="sigmoid"))
        if encoder_dropout != 0:
            encoder_L.add(TimeDistributed(Dropout(encoder_dropout)))
        encoder_L.add(MaxPooling1D(pool_length=LSTM_size))
        encoder_L.add(Flatten())

    # without maxpooling
    else:
        encoder_L.add(Masking(mask_value=0.))
        encoder_L.add(LSTM(LSTM_size, return_sequences=False, inner_activation="sigmoid"))
        if encoder_dropout != 0:
            encoder_L.add(Dropout(encoder_dropout))

    # encode the second entity
    encoder_R = Sequential()

    encoder_R.add(Dropout(input_dropout, input_shape=(data_dim, max_len)))

    # with maxpooling
    if maxpooling:
        encoder_R.add(LSTM(LSTM_size, return_sequences=True, inner_activation="sigmoid"))
        if encoder_dropout != 0:
            encoder_R.add(TimeDistributed(Dropout(encoder_dropout)))
        encoder_R.add(MaxPooling1D(pool_length=LSTM_size))
        encoder_R.add(Flatten())

    else:
    # without maxpooling
        encoder_R.add(Masking(mask_value=0.))
        encoder_R.add(LSTM(LSTM_size, return_sequences=False, inner_activation="sigmoid"))
        if encoder_dropout != 0:
            encoder_R.add(Dropout(encoder_dropout))

    # combine and classify entities as a single relation
    decoder = Sequential()
    decoder.add(Merge([encoder_R, encoder_L], mode='concat'))
    decoder.add(Dense(dense_size, W_regularizer=W_reg, b_regularizer=B_reg, activity_regularizer=act_reg, activation='sigmoid'))
    if decoder_dropout != 0:
        decoder.add(Dropout(decoder_dropout))
    decoder.add(Dense(nb_classes, W_regularizer=W_reg, b_regularizer=B_reg, activity_regularizer=act_reg, activation='softmax'))

    # compile the final model
    decoder.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return decoder

def train_model(notes, model=None, epochs=5, training_input=None, weight_classes=False, batch_size=256,
    encoder_dropout=0, decoder_dropout=0, input_dropout=0, reg_W=0, reg_B=0, reg_act=0, LSTM_size=32, dense_size=100, maxpooling=True, data_dim=300, max_len='auto', nb_classes=19):
    '''
    obtains entity pairs and tlink labels from every note passed, and uses them to train the network.
    params:
        notes: timeML files to train on
        epochs: number of training epochs to perform
        training_input: provide processed training matrices directly, rather than rebuilding the matrix from notes every time. Useful for training multipul models with the same data.
            Formatted as a tuple; (XL, XR, labels)
        weight_classes: rather or not to use class weighting
        batch_size: size of training batches to use
        max_len: either an integer specifying the maximum input sequence length, or 'auto', which infer maximum length from the training data
    All other parameters feed directly into get_untrained_model(), and are described there.
    '''
    if training_input == None:
        XL, XR, labels = _get_training_input(notes)
        print "training input obtained..."
    else:
        XL, XR, labels = training_input

    # reformat labels so that they can be used by the NN
    Y = to_categorical(labels,nb_classes)
    print "labels reformed..."

    # use weighting to assist with the imbalanced data set problem
    if weight_classes:
        class_weights = get_uniform_class_weights(Y)
    else:
        class_weights = None

    # infer maximum sequence length
    if max_len == 'auto':
        max_len = XL.shape[2]
    # pad input to reach max_len
    else:
        filler = np.ones((1,1,max_len))
        XL, _ = _pad_to_match_dimensions(XL, filler, 2, pad_left=True)
        XR, _ = _pad_to_match_dimensions(XR, filler, 2, pad_left=True)

    if model is None:
        model = get_untrained_model(encoder_dropout=encoder_dropout, decoder_dropout=decoder_dropout, input_dropout=input_dropout, reg_W=reg_W, reg_B=reg_B, reg_act=reg_act, LSTM_size=LSTM_size, dense_size=dense_size,
        maxpooling=maxpooling, data_dim=data_dim, max_len=max_len, nb_classes=nb_classes)
    else:
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # split off validation data with 20 80 split (this way we get the same validation data every time we use this data sample, and can test on it after to get a confusion matrix)
    V_XL = XL[:(XL.shape[0]/5),:,:]
    V_XR = XR[:(XR.shape[0]/5),:,:]
    V_Y  = Y [:( Y.shape[0]/5),:]
    V_labels = labels[:(Y.shape[0]/5)]

    XL = XL[(XL.shape[0]/5):,:,:]
    XR = XR[(XR.shape[0]/5):,:,:]
    Y  = Y [( Y.shape[0]/5):,:]

    # train the network
    print 'Training network...'
    training_history = model.fit([XL, XR], Y, nb_epoch=epochs, validation_split=0.2, class_weight=class_weights, batch_size=batch_size, validation_data=([V_XL, V_XR], V_Y))
    json.dump(training_history.history, open('training_history.json', 'w'))
    test = model.predict_classes([V_XL, V_XR])

    print test
    class_confusion(test, V_labels, nb_classes)

    return model

def predict(notes, detector, classifier, evalu=False):
    '''
    using the given detector and classifier, predict labels for all pairs in the given note set
    '''

    XL, XR, del_lists, gold_labels = _get_test_input(notes, evaluation=evalu)

    # get expected length of model input
    input_len = detector.input_shape[0][2]
    filler = np.ones((1,1,input_len))

    # pad input matrix to fit expected length
    XL, _ = _pad_to_match_dimensions(XL, filler, 2, pad_left=True)
    XR, _ = _pad_to_match_dimensions(XR, filler, 2, pad_left=True)

    print 'Detecting...'
    presense_labels = detector.predict_classes([XL, XR])

    # get data tensors for classification
    C_XL = XL[presense_labels[:]==1]
    C_XR = XR[presense_labels[:]==1]

    # if input is too long, cut off right side terms
    # TODO: cut left terms instead
    input_len = classifier.input_shape[0][2]

    C_XL = _strip_to_length(C_XL, input_len, 2)
    C_XR = _strip_to_length(C_XR, input_len, 2)

    print C_XL.shape

    print 'Classifying...'
    classification_labels = classifier.predict_classes([C_XL, C_XR])

    # combine classification into final label list
    labels = []
    class_index = 0
    for label in presense_labels:
        if label == 1:
            labels.append(classification_labels[class_index])
            class_index += 1
        else:
            labels.append(0)

    assert len(labels) == len(presense_labels)

    if evalu:
        class_confusion(labels, gold_labels)

    return _convert_int_labels_to_str(labels), del_lists

def single_predict(notes, model, evalu=False):
    '''
    predict using a trained single pass model
    '''

    XL, XR, del_lists, gold_labels = _get_test_input(notes, evaluation=evalu)

    # get expected length of model input
    input_len = model.input_shape[0][2]
    filler = np.ones((1,1,input_len))

    # pad input matrix to fit expected length
    XL, _ = _pad_to_match_dimensions(XL, filler, 2, pad_left=True)
    XR, _ = _pad_to_match_dimensions(XR, filler, 2, pad_left=True)

    print 'Predicting...'
    labels = model.predict_classes([XL, XR])

    if evalu:
        class_confusion(labels, gold_labels, 6)

    return _convert_int_labels_to_str(labels), del_lists

def _get_training_input(notes, no_none=False, presence=False, shuffle=True):

    # TODO: handle tlinks linking to the document creation time. at the moment, we simply skip them

    # labels for each SDP pair
    semlinklabels = []

    # data tensor for left and right SDP subpaths
    XL = None
    XR = None

    print 'Loading word embeddings...'
    word_vectors = load_word2vec_binary(os.environ["TEA_PATH"]+'/GoogleNews-vectors-negative300.bin', verbose=0)
    # word_vectors = load_word2vec_binary(os.environ["TEA_PATH"]+'/wiki.dim-300.win-8.neg-15.skip.bin', verbose=0)

    print 'Extracting dependency paths...'
    for i, note in enumerate(notes):
        # get tlink lables
        note_semlinklabels = note.get_semlink_labels()

        # get the representation for the event/timex pairs in the note
        # will be 3D tensor with axis zero holding the each pair, axis 1 holding the word embeddings
        # (with length equal to word embedding length), and axis 2 hold each word.
        # del_list is a list of indices for which no SDP could be obtained
        left_vecs, right_vecs, del_list = _extract_path_representations(note, word_vectors, no_none)

        # add the note's data to the combine data matrix
        if XL == None:
            XL = left_vecs
        else:
            XL = _pad_and_concatenate(XL, left_vecs, axis=0, pad_left=[2])

        if XR == None:
            XR = right_vecs
        else:
            XR = _pad_and_concatenate(XR, right_vecs, axis=0, pad_left=[2])

        # remove duplicate indices
        del_list = list(set(del_list))
        # remove indices in descending order so that they continue to refer to the item we want to remove
        del_list.sort()
        del_list.reverse()
        for index in del_list:
            del note_semlinklabels[index]

        # add remaining labels to complete list of labels
        semlinklabels += note_semlinklabels

    # pad XL and XR so that they have the same number of dimensions on the second axis
    # any other dimension mis-matches are caused by actually errors and should not be padded away
    XL, XR = _pad_to_match_dimensions(XL, XR, 2, pad_left=True)

    # # extract longest input sequence in the training data, and ensure both matrices
    # input_len = XL.shape[2]

    labels = _convert_str_labels_to_int(semlinklabels)

    if presence:
        for i, label in enumerate(labels):
            if label != 0:
                labels[i] = 1

    if shuffle:
        rng_state = np.random.get_state()
        np.random.shuffle(XL)
        np.random.set_state(rng_state)
        np.random.shuffle(XR)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

    return XL, XR, labels

def _get_test_input(notes, evaluation=False):
    '''
    get input tensors for prediction. If evaluation is true, gold labels are also extracted
    '''

    # data tensors for left and right SDP
    XL = None
    XR = None

    # list of lists, where each list contains the indices to delete from a given note
    # indices of the outer list corespond to indices of the notes list
    del_lists = []

    tlinklabels = []

    print 'Loading word embeddings...'
    word_vectors = load_word2vec_binary(os.environ["TEA_PATH"]+'/GoogleNews-vectors-negative300.bin', verbose=0)
    # word_vectors = load_word2vec_binary(os.environ["TEA_PATH"]+'/wiki.dim-300.win-8.neg-15.skip.bin', verbose=0)

    print 'Extracting dependency paths...'
    for i, note in enumerate(notes):
        # get the representation for the event/timex pairs in the note
        # will be 3D tensor with axis zero holding the each pair, axis 1 holding the word embeddings
        # (with length equal to word embedding length), and axis 2 hold each word.
        left_vecs, right_vecs, del_list = _extract_path_representations(note, word_vectors)

        # add the note's data to the combine data matrix
        if XL == None:
            XL = left_vecs
        else:
            XL = _pad_and_concatenate(XL, left_vecs, axis=0, pad_left=[2])

        if XR == None:
            XR = right_vecs
        else:
            XR = _pad_and_concatenate(XR, right_vecs, axis=0, pad_left=[2])

        # remove duplicate indices
        del_list = list(set(del_list))
        # remove indices in descending order so that they continue to refer to the item we want to remove
        del_list.sort()
        del_list.reverse()
        del_lists.append(del_list)

        if evaluation:
            # get tlink lables
            note_tlinklabels = note.get_tlink_labels()
            for index in del_list:
                del note_tlinklabels[index]
            tlinklabels += note_tlinklabels

    # pad XL and XR so that they have the same number of dimensions on the second axis
    # any other dimension mis-matches are caused by actually errors and should not be padded away
    XL, XR = _pad_to_match_dimensions(XL, XR, 2, pad_left=True)

    labels = []
    if evaluation:
        labels = _convert_str_labels_to_int(tlinklabels)

    return XL, XR, del_lists, labels

def _extract_path_representations(note, word_vectors, no_none=False):
    '''
    convert a note into a portion of the input matrix
    '''

    # retrieve the ids for the left and right halves of every SDP between tlinked entities
    left_ids, right_ids = _get_token_id_subpaths(note)
    semlinklabels = _convert_str_labels_to_int(note.relations)

    # left and right paths are used to store the actual tokens of the SDP paths
    left_paths = []
    right_paths = []

    # del list stores the indices of pairs which do not have a SDP so that they can be removed from the labels later
    del_list = []

    # get token text from ids in left sdp
    for i, id_list in enumerate(left_ids):
        if semlinklabels[i] == 0 and no_none:
           left_paths.append([])
        else:
            left_paths.append(note.get_tokens_from_ids(id_list)) # list of list of words

    # get token text from ids in right sdp
    for i, id_list in enumerate(right_ids):
        if semlinklabels[i] == 0 and no_none:
           right_paths.append([])
        else:
            right_paths.append(note.get_tokens_from_ids(id_list))

    # get the word vectors for every word in the left path
    left_vecs = None
    for j, path in enumerate(left_paths):
        vecs_path = None
        for word in path:
            # try to get embedding for a given word. If the word is not in the vocabulary, use a vector of all 1s.
            try:
                embedding = word_vectors[word]
            except KeyError:
                embedding = np.random.uniform(low=-0.5, high=0.5, size=(300))

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
            left_vecs = _pad_and_concatenate(left_vecs, vecs_path, axis=0, pad_left=[2])

    # get the vectors for every word in the right path
    right_vecs = None
    for j, path in enumerate(right_paths):
        vecs_path = None
        for word in path:
            # try to get embedding for a given word. If the word is not in the vocabulary, use a vector of all 0s.
            try:
                embedding = word_vectors[word]
            except KeyError:
                embedding = np.random.uniform(low=-0.5, high=0.5, size=(300))

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
            right_vecs = _pad_and_concatenate(right_vecs, vecs_path, axis=0, pad_left=[2])

    return left_vecs, right_vecs, del_list

def _pad_and_concatenate(a, b, axis, pad_left=[]):
    '''
    given two tensors, pad with zeros so that all dimensions are equal except the axis to concatentate on, and the concatenate
    pad_left is a list of dimensions to pad the left side on. Other dimensions will recieve right sided padding
    '''
    # if tensors have number of dimensions
    if len(a.shape) != len(b.shape):
        pass

    # for each axis that is not to be concatenated on, pad the smaller of the two tensors with zeros so that both have equal sizes
    for i in range(len(a.shape)):
        if i == axis:
            continue
        if i in pad_left:
            a, b = _pad_to_match_dimensions(a, b, i, pad_left=True)
        else:
            a, b = _pad_to_match_dimensions(a, b, i)

    # concatenate and return the padded matrices
    return np.concatenate((a, b), axis=axis)

def _pad_to_match_dimensions(a, b, axis, pad_left=False):
    '''
    given to tensors and an axis for comparison, pad the smaller of the two with zeros to match the dimensions of the larger
    '''

    # get function to pad correct side
    if pad_left:
        concat = lambda X, _pad_shape, _axis: np.concatenate((np.zeros(tuple(_pad_shape)), X), axis=_axis)
    else:
        concat = lambda X, _pad_shape, _axis: np.concatenate((X, np.zeros(tuple(_pad_shape))), axis=_axis)

    a_axis = a.shape[axis]
    b_axis = b.shape[axis]
    if a_axis > b_axis:
        pad_shape = list(b.shape)
        pad_shape[axis] = a_axis - b_axis
        b = concat(b, pad_shape, axis)
    if b_axis > a_axis:
        pad_shape = list(a.shape)
        pad_shape[axis] = b_axis - a_axis
        a = concat(a, pad_shape, axis)

    return a, b

def _strip_to_length(a, length, axis):

    if a.shape[axis] > length:
        snip = a.shape[axis] - length
        a = a[:,:,snip:]

    return a

def _get_token_id_subpaths(note):
    '''
    extract ids for the tokens in each half of the shortest dependency path between each token in each relation
    '''
    # TODO: for now we only look at the first token in a given entity. Eventually, we should get all tokens in the entity

    pairs = note.semLinks

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

        left_path, right_path = note.dependency_paths.get_left_right_subpaths(source_id, target_id)
        left_paths.append(left_path)
        right_paths.append(right_path)

    return left_paths, right_paths

def get_uniform_class_weights(labels):
    '''
    get a dictionary of weights for each class. Used to combat imbalanced data problems by reducing
    the impact of highly represented classes. Has no effect on equally distributed data
    labels is composed of one-hot vectors.
    '''
    n_samples = len(labels)
    n_classes = len(labels[0])

    weights = n_samples/ (n_classes * np.sum(labels, axis=0))

    return weights
def _convert_str_labels_to_int(labels):
    '''
    convert tlink labels to integers so they can be processed by the network
    '''

    global ignore_order
    print "ingore_order", ignore_order

    processed_labels = []
    for label in labels:
        if label == "Cause-Effect(e1,e2)":
            processed_labels.append(1)
        elif label == "Instrument-Agency(e1,e2)":
            processed_labels.append(2)
        elif label == "Product-Producer(e1,e2)":
            processed_labels.append(3)
        elif label == "Content-Container(e1,e2)":
            processed_labels.append(4)
        elif label == "Entity-Origin(e1,e2)":
            processed_labels.append(5)
        elif label == "Entity-Destination(e1,e2)":
            processed_labels.append(6)
        elif label == "Component-Whole(e1,e2)":
            processed_labels.append(7)
        elif label == "Member-Collection(e1,e2)":
            processed_labels.append(8)
        elif label == "Message-Topic(e1,e2)":
            processed_labels.append(9)
        elif label == "Other":
            processed_labels.append(10)
        elif label == "Cause-Effect(e2,e1)":
            processed_labels.append(11)
        elif label == "Instrument-Agency(e2,e1)":
            processed_labels.append(12)
        elif label == "Product-Producer(e2,e1)":
            processed_labels.append(13)
        elif label == "Content-Container(e2,e1)":
            processed_labels.append(14)
        elif label == "Entity-Origin(e2,e1)":
            processed_labels.append(15)
        elif label == "Entity-Destination(e2,e1)":
            processed_labels.append(16)
        elif label == "Component-Whole(e2,e1)":
            processed_labels.append(17)
        elif label == "Member-Collection(e2,e1)":
            processed_labels.append(18)
        elif label == "Message-Topic(e2,e1)":
            processed_labels.append(19)
        else:
            print "unkonwn class:", label
        #     processed_labels.append(0)

    if ignore_order:
        processed_labels = [x%10 for x in processed_labels]
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
            processed_labels.append("IS_INCLUDED")
        elif label == 4:
            processed_labels.append("BEGUN_BY")
        elif label == 5:
            processed_labels.append("ENDED_BY")
        else:  # label for unlinked pairs (should have int 0)
            processed_labels.append("None")

    return processed_labels

def class_confusion(predicted, actual, nb_classes):
    '''
    print confusion matrix for two lists of labels. A given index in both lists should correspond to the same data sample
    '''

    confusion = np.zeros((nb_classes, nb_classes), dtype='int16')

    for true, pred in zip(actual, predicted):
        # build confusion matrix
        confusion[true, pred] += 1
    pickle.dump(confusion, open('evaluation.pkl', 'w'))
    # print confusion matrix
    print "confusion matrix"
    print "rows: actual labels.  columns: predicted labels."
    for i, row in enumerate(confusion):
        print i, ": ", row

    print classification_report(actual, predicted)

    # TODO: add F-1, precision, recall for each class

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
    # test.model.fit([input1,input2], _labels, nb_epoch=100)
    # print test.model.predict_classes([input1,input2])
    # print labels
    pass
