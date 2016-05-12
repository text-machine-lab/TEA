import os
import features
import cPickle
import sys

TEA_HOME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

from sci import train as train_classifier

# models to be loaded.
# not be manually set.
_models = {}
_vects = {}

_models_loaded = False

def train(notes, train_timex=True, train_event=True, train_rel=True):

    # TODO: need to do some filtering of tokens
    # TODO: experiment with the feature of the 4 left and right taggings. do we
    #       only utilize taggings for each pass or do we incorporate taggings in different passes?

    timexLabels   = [] # BIO labelings for tokens in text.
    timexFeatures = []

    eventLabels   = [] # EVENT or O labelings for tokens in text.
    eventFeatures = []

    eventClassLabels   = [] # event class labelings for tokens in text.
    eventClassFeatures = []

    tlinkLabels   = {} # temporal relation labelings for enitity pairs.
    tlinkFeatures = []

    timexClassifier = None
    timexVectorizer = None

    eventClassifier = None
    eventVectorizer = None

    eventClassClassifier = None
    eventClassVectorizer = None

    tlinkClassifier = None
    tlinkVectorizer = None

    for i, note in enumerate(notes):

        print "note: {}".format(i)

        if train_timex is True:
            # extract features to perform BIO labeling for timexs
            tmpLabels = note.get_timex_labels()
            for label in tmpLabels: timexLabels += label
            timexFeatures += features.extract_timex_feature_set(note, tmpLabels)

        if train_event is True:
            # extract features to perform EVENT or O labeling.
            tmpLabels = note.get_event_labels()
            for label in tmpLabels: eventLabels += label
            eventFeatures += features.extract_event_feature_set(note, tmpLabels)

            # extract features to perform event class labeling.
            tmpLabels = note.get_event_class_labels()
            for label in tmpLabels: eventClassLabels += label
            eventClassFeatures += features.extract_event_class_feature_set(note, tmpLabels, note.get_event_labels())

        if train_rel is True:
            # extract features to classify relations between temporal entities.
            tlinkLabels = note.get_tlink_labels()
            tlinkFeatures += features.extract_tlink_features(note)

    # TODO: when predicting, if gold standard is provided evaluate F-measure for each of the steps

    if train_timex is True:
        # train model to perform BIO labeling for timexs
        timexClassifier, timexVectorizer = _trainTimex(timexFeatures, timexLabels, grid=True)
        #timexClassifier, timexVectorizer = _trainTimex(timexFeatures, timexLabels, grid=False)

    if train_event is True:
        # train model to label as EVENT or O
        # TODO: filter non-timex only?
        eventClassifier, eventVectorizer = _trainEvent(eventFeatures, eventLabels, grid=True)
       # eventClassifier, eventVectorizer = _trainEvent(eventFeatures, eventLabels, grid=False)

        # train model to label as a class of EVENT
        # TODO: filter event only?
        # TODO: should we be training over all tokens or those that are just EVENTs?
        eventClassClassifier, eventClassVectorizer = _trainEventClass(eventClassFeatures, eventClassLabels, grid=True)
        #eventClassClassifier, eventClassVectorizer = _trainEventClass(eventClassFeatures, eventClassLabels, grid=False)

    if train_rel is True:
        # train model to classify relations between temporal entities.
        # TODO: add features back in.
        #tlinkClassifier, tlinkVectorizer = _trainTlink(tlinkFeatures, tlinkLabels, grid=False)
        tlinkClassifier, tlinkVectorizer = _trainTlink(tlinkFeatures, tlinkLabels, grid=True)

    # will be accessed later for dumping
    models = {"TIMEX":timexClassifier,
              "EVENT":eventClassifier,
              "EVENT_CLASS":eventClassClassifier,
              "TLINK":tlinkClassifier}

    vectorizers = {"TIMEX":timexVectorizer,
                   "EVENT":eventVectorizer,
                   "EVENT_CLASS":eventClassVectorizer,
                   "TLINK":tlinkVectorizer}

    return models, vectorizers


def predict(note, predict_timex=True, predict_event=True, predict_rel=True):

    # TODO: try and correct the flattening on the lists. might just end up being redundent?
    # TODO: refactor this code. a lot of it is redundant.
    # TODO: need to do some filtering of tokens
    # TODO: experiment with the feature of the 4 left and right taggings. do we
    #       only utilize taggings for each pass or do we incorporate taggings in different passes?

    global _models
    global _vects
    global _models_loaded

    if _models_loaded is False:
        sys.exit("Models not loaded. Cannot predict")

    # get tokenized text
    tokenized_text = note.get_tokenized_text()

    # will be new iob_labels
    iob_labels      = []

    timexLabels      = []
    eventLabels      = []
    eventClassLabels = []

    tlink_labels = []

    # init the number of lines for timexlabels
    # we currently do not know what they are.
    # get the tokens into a flast list, these are ordered by
    # appearance within the document
    tokens = []
    for line in tokenized_text:
        timexLabels.append([])
        eventLabels.append([])
        eventClassLabels.append([])
        iob_labels.append([])
        tokens += tokenized_text[line]

    if predict_timex is True:

        timexClassifier = _models["TIMEX"]
        timexVectorizer = _vects["TIMEX"]

        # get the timex feature set for the tokens within the note.
        timexFeatures = features.extract_timex_feature_set(note, timexLabels, predict=True)

        # sanity check
        assert len(tokens) == len(timexFeatures)

        # predict over the tokens and the features extracted.
        for t, f in zip(tokens, timexFeatures):

            features.update_features(t, f, timexLabels)

            X = timexVectorizer.transform([f]).toarray()
            Y = list(timexClassifier.predict(X))

            timexLabels[t["sentence_num"] - 1].append({'entity_label':Y[0],
                                                       'entity_type':None if Y[0] == 'O' else 'TIMEX3',
                                                       'entity_id':None})

            iob_labels[t["sentence_num"] - 1].append(timexLabels[t["sentence_num"] - 1][-1])

    if predict_event is True:

        eventClassifier = _models["EVENT"]
        eventVectorizer = _vects["EVENT"]

        eventClassClassifier = _models["EVENT_CLASS"]
        eventClassVectorizer = _vects["EVENT_CLASS"]

        # get the timex feature set for the tokens within the note.
        # don't get iob labels yet, they are inaccurate. need to predict first.
        eventFeatures = features.extract_event_feature_set(note, eventLabels, predict=True)

        # sanity check
        assert len(tokens) == len(eventFeatures)

        # TODO: need to do some filter. if something is already labeled then just skip over it.
        # predict over the tokens and the features extracted.
        for t, f in zip(tokens, eventFeatures):

            features.update_features(t, f, eventLabels)

            X = eventVectorizer.transform([f]).toarray()
            Y = list(eventClassifier.predict(X))

            eventLabels[t["sentence_num"] - 1].append({'entity_label':Y[0],
                                                       'entity_type':None if Y[0] == 'O' else 'EVENT',
                                                       'entity_id':None})

        # get the timex feature set for the tokens within the note.
        eventClassFeatures = features.extract_event_class_feature_set(note, eventClassLabels, eventLabels, predict=True)

        # sanity check
        assert len(tokens) == len(eventClassFeatures)

        # predict over the tokens and the features extracted.
        for t, f in zip(tokens, eventClassFeatures):

            # updates labels
            features.update_features(t, f, eventClassLabels)

            X = eventClassVectorizer.transform([f]).toarray()
            Y = list(eventClassClassifier.predict(X))

            eventClassLabels[t["sentence_num"] - 1].append({'entity_label':Y[0],
                                                            'entity_type':None if Y[0] == 'O' else 'EVENT',
                                                            'entity_id':None})

            if iob_labels[t["sentence_num"] - 1][t["token_offset"]]["entity_type"] == None:
                iob_labels[t["sentence_num"] - 1][t["token_offset"]] = eventClassLabels[t["sentence_num"] - 1][-1]

    _totLabels = []
    for l in eventClassLabels:
        _totLabels += l

    print "predicted ZERO events? : ", len(_totLabels) == len([l for l in _totLabels if l["entity_label"] != 'O'])

    _totLabels = []
    for l in timexLabels:
        _totLabels += l

    print "predicted ZERO timex? :", len(_totLabels) == len([l for l in _totLabels if l["entity_label"] != 'O'])

    if predict_timex is True and predict_event is True and predict_rel is True:

        tlinkVectorizer = _vects["TLINK"]
        tlinkClassifier = _models["TLINK"]

        note.set_tlinked_entities(timexLabels,eventClassLabels)
        note.set_iob_labels(iob_labels)

        print "PREDICT: getting tlink features"

        f = features.extract_tlink_features(note)
        X = tlinkVectorizer.transform(f).toarray()

        tlink_labels = list(tlinkClassifier.predict(X))

    entity_labels    = [label for line in iob_labels for label in line]
    original_offsets = note.get_token_char_offsets()

    return entity_labels, original_offsets, tlink_labels, tokens


def _trainTimex(timexFeatures, timexLabels, grid=False):
    """
    Purpose: Train a classifer for Timex3 expressions

    @param tokenVectors: A list of tokens represented as feature dictionaries
    @param Y: A list of lists of Timex3 classifications for each token in each sentence
    """

    assert len(timexFeatures) == len(timexLabels), "{} != {}".format(len(timexFeatures), len(timexLabels))

    Y = [l["entity_label"] for l in timexLabels]

    clf, vec = train_classifier(timexFeatures, Y, do_grid=grid, ovo=True)
    return clf, vec


def _trainEvent(eventFeatures, eventLabels, grid=False):
    """
    Model::_trainEvent()

    Purpose: Train a classifer for event identification

    @param tokenVectors: A list of tokens represented as feature dictionaries
    @param Y: A list of lists of event classifications for each token, with one list per sentence
    """

    assert len(eventFeatures) == len(eventLabels), "{} != {}".format(len(eventFeatures), len(eventLabels))

    Y = [l["entity_label"] for l in eventLabels]

    clf, vec = train_classifier(eventFeatures, Y, do_grid=grid)
    return clf, vec


def _trainEventClass(eventClassFeatures, eventClassLabels, grid=False):
    """
    Model::_trainEventClass()

    Purpose: Train a classifer for event class identification

    @param tokenVectors: A list of tokens represented as feature dictionaries
    @param Y: A list of lists of event classifications for each token, with one list per sentence
    """

    assert len(eventClassFeatures) == len(eventClassLabels), "{} != {}".format(len(eventClassFeatures), len(eventClassLabels))

    Y = [l["entity_label"] for l in eventClassLabels]

    clf, vec = train_classifier(eventClassFeatures, Y, do_grid=grid)
    return clf, vec


def _trainTlink(tokenVectors, Y, grid=False):
    """
    Model::_trainRelation()

    Purpose: Train a classifer for temporal relations between events and timex3 labels

    @param tokenVectors: A list of tokens represented as feature dictionaries
    @param Y: A list of relation classifications for each pair of timexes and events.
    """

    assert len(tokenVectors) == len(Y)

    clf, vec = train_classifier(tokenVectors, Y, do_grid=grid, ovo=True)
    return clf, vec


def combineLabels(timexLabels, eventLabels, OLabels=[]):
    """
    combineTimexEventLabels():
        merge event and timex labels into one list, adding instance ids

    @param timexLabels: list of timex labels for entities.
    @param eventLabels: list of event labels for entities. Includes no instances labeled as timexs
    @return: list of dictionaries, with one dictionary for each entity
    """

    labels = []

    # creation time is always t0
    for i, timexLabel in  enumerate(timexLabels):
        label = {"entity_label": timexLabel, "entity_type": "TIMEX3", "entity_id": "t" + str(i+1)}
        labels.append(label)

    for i, eventLabel in enumerate(eventLabels):
        label = {"entity_label": eventLabel, "entity_type": "EVENT", "entity_id": "e" + str(i)}
        labels.append(label)

    for i, Olabel in enumerate(OLabels):
        label = {"entity_label": Olabel, "entity_type": None, "entity_id": None}
        labels.append(label)

    assert len(labels) == len(timexLabels + eventLabels + OLabels)

    return labels

def load_models(path, predict_timex, predict_event, predict_tlink):

    keys = ["TIMEX", "EVENT", "EVENT_CLASS", "TLINK"]
    flags = [predict_timex, predict_event, predict_event, predict_tlink]

    global _models
    global _vects
    global _models_loaded

    for key, flag in zip(keys, flags):

        # vect should also exist, unless something went wrong.
        if os.path.isfile(path+"_"+key+"_MODEL") is True and flag is True:
            print "loading: {}".format(key)

            _models[key] = cPickle.load(open(path+"_"+key+"_MODEL", "rb"))
            _vects[key]  = cPickle.load(open(path+"_"+key+"_VECT", "rb"))
        else:
            _models[key] = None
            _vects[key]  = None

    _models_loaded = True

    return


def dump_models(models, vectorizers, path):
    """dump model specified by argument into the file path indicated by path argument
    """

    print "dumping..."

    keys = ["TIMEX", "EVENT", "EVENT_CLASS", "TLINK"]

    for key in keys:
        if models[key] is None:
            continue
        else:

            print "dumping: {}".format(key)

            model_dest = open(path+"_"+key+"_MODEL", "wb")
            vect_dest  = open(path+"_"+key+"_VECT", "wb")

            cPickle.dump(models[key], model_dest)
            cPickle.dump(vectorizers[key], vect_dest)

    return

