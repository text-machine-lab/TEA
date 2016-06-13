import os
import features
import cPickle
import sys

TEA_HOME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

sys.path.insert(0,TEA_HOME_DIR)

from code.notes.utilities.heidel_time_util import get_iobs_heidel

from sci import train as train_classifier

# models to be loaded.
# not be manually set.
_models = {}
_vects = {}

_models_loaded = False

def train(notes, train_event=True, train_rel=True, predicate_as_event=False):

    # TODO: need to do some filtering of tokens
    # TODO: experiment with the feature of the 4 left and right taggings. do we
    #       only utilize taggings for each pass or do we incorporate taggings in different passes?

    eventLabels   = [] # EVENT or O labelings for tokens in text.
    eventFeatures = []

    eventClassLabels   = [] # event class labelings for tokens in text.
    eventClassFeatures = []

    tlinkLabels   = [] # temporal relation labelings for enitity pairs.
    tlinkFeatures = []

    eventClassifier = None
    eventVectorizer = None

    eventClassClassifier = None
    eventClassVectorizer = None

    tlinkClassifier = None
    tlinkVectorizer = None

    for i, note in enumerate(notes):

        print "note: {}".format(i)
        print "note path: ", note.note_path

        if train_event is True:

            if predicate_as_event is False:
                # extract features to perform EVENT or O labeling.
                tmpLabels = note.get_event_labels()
                for label in tmpLabels: eventLabels += label
                eventFeatures += features.extract_event_feature_set(note, tmpLabels, timexLabels=note.get_timex_labels())

            #print "predicate_as_event: ", predicate_as_event

            if predicate_as_event:

             #   print "PREDICATE AS EVENT FEAT EXTRACT"

                _eventClassLabels = []

                # extract features to perform event class labeling.
                tmpLabels = note.get_event_class_labels()
                for label in tmpLabels: _eventClassLabels += label
                _eventClassFeatures = features.extract_event_class_feature_set(note, tmpLabels, note.get_event_labels(), timexLabels=note.get_timex_labels())

                tokenized_text = note.get_tokenized_text()
                tokens = [token for line in tokenized_text for token in tokenized_text[line]]

                assert len(tokens) == len(_eventClassLabels)
                assert len(tokens) == len(_eventClassFeatures)

                for token, label, f in zip(tokens, _eventClassLabels, _eventClassFeatures):
                    if token["is_predicate"]:
                        eventClassLabels.append(label)
                        eventClassFeatures.append(f)
            else:
                # extract features to perform event class labeling.
                tmpLabels = note.get_event_class_labels()
                for label in tmpLabels: eventClassLabels += label
                eventClassFeatures += features.extract_event_class_feature_set(note, tmpLabels, note.get_event_labels(), timexLabels=note.get_timex_labels())

        if train_rel is True:
            # extract features to classify relations between temporal entities.
            tlinkLabels += note.get_tlink_labels()
            tlinkFeatures += features.extract_tlink_features(note, event_labels=note.get_event_class_labels(), timex_labels=note.get_timex_labels())

    # TODO: when predicting, if gold standard is provided evaluate F-measure for each of the steps

    #print "VOCAB:"
    #print features._voc
    #print

    if train_event is True:

        if predicate_as_event is False:
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
    models = {"EVENT":eventClassifier,
              "EVENT_CLASS":eventClassClassifier,
              "TLINK":tlinkClassifier}

    vectorizers = {"EVENT":eventVectorizer,
                   "EVENT_CLASS":eventClassVectorizer,
                   "TLINK":tlinkVectorizer}

    return models, vectorizers

def eval_tlink_predict(note, predict_timex=True, predict_event=True, predict_rel=True, predicate_as_event=False):

    print "called eval_tlink_predict"
    global _models
    global _vects
    global _models_loaded

    if _models_loaded is False:
        sys.exit("Models not loaded. Cannot predict")

    note.iob_labels = []
    iob_labels = note.get_labels()

    tlinkVectorizer = _vects["TLINK"]
    tlinkClassifier = _models["TLINK"]

    event_labels = []
    timex_labels = []
    tokens = []

    _eventLabels = []
    _timexLabels = []

    tokenized_text = note.get_tokenized_text()

    _iob_labels = []

    for line in note.get_event_class_labels():
        event_labels += line
    for line in note.get_timex_labels():
        timex_labels += line
    for line in tokenized_text:
        tokens += tokenized_text[line]

    for line in tokenized_text:
        _timexLabels.append([])
        _eventLabels.append([])
        _iob_labels.append([])

    i = 0

    for t, timex_label, event_label in zip(tokens, timex_labels, event_labels):

        # print timex_label

        _timexLabels[t["sentence_num"] - 1].append({'entity_label':timex_label["entity_label"],
                                                   'entity_type':None if timex_label["entity_label"] == 'O' else 'TIMEX3',
                                                   'entity_id':timex_label["entity_id"],
                                                   'norm_val':iob_labels[t["sentence_num"]-1][t["token_offset"]]["norm_val"]})

        _iob_labels[t["sentence_num"]-1].append(_timexLabels[t["sentence_num"]-1][-1])

    for t, timex_label, event_label in zip(tokens, timex_labels, event_labels):

        _eventLabels[t["sentence_num"]-1].append({'entity_label':event_label["entity_label"],
                                                  'entity_type':None if event_label["entity_label"] == 'O' else 'EVENT',
                                                  'entity_id':event_label["entity_id"]})

        if _iob_labels[t["sentence_num"]-1][t["token_offset"]]["entity_type"] == None:
            _iob_labels[t["sentence_num"]-1][t["token_offset"]] = _eventLabels[t["sentence_num"]-1][-1]

    note.tlinks = []
    note.set_tlinked_entities(_timexLabels, _eventLabels, use_gold_ids=True)
    note.iob_labels = []

#    print "calling set_iob_labels"
#    print "_iob_labels: "
#    print _iob_labels
#    print

    note.set_iob_labels(_iob_labels)

    f = features.extract_tlink_features(note, event_labels=_eventLabels, timex_labels=_timexLabels)
    X = tlinkVectorizer.transform(f)

    tlink_labels = list(tlinkClassifier.predict(X))

    entity_labels    = [label for line in note.get_labels() for label in line]
    original_offsets = note.get_token_char_offsets()

    return entity_labels, original_offsets, tlink_labels, tokens


def predict(note, predict_timex=True, predict_event=True, predict_rel=True, predicate_as_event=False):

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

    timex_count = 2

    if predict_timex is True:

        heidel_iobs = []
        for line in get_iobs_heidel(note):
            heidel_iobs += line

        assert len(heidel_iobs) == len(tokens)

        # predict over the tokens and the features extracted.
        for t, l in zip(tokens, heidel_iobs):

            timexLabels[t["sentence_num"] - 1].append({'entity_label':l["entity_label"],
                                                       'entity_type':None if l["entity_label"] == 'O' else 'TIMEX3',
                                                       'entity_id':"t"+str(timex_count),
                                                       'norm_val':l["norm_val"]})
            timex_count += 1
            iob_labels[t["sentence_num"] - 1].append(timexLabels[t["sentence_num"] - 1][-1])

    else:
        for t in tokens:
            timexLabels[t["sentence_num"] - 1].append({'entity_label':'O',
                                                      'entity_type':None,
                                                      'entity_id':"t"+str(timex_count),
                                                      'norm_val':None})
            timex_count += 1
            iob_labels[t["sentence_num"] - 1].append(timexLabels[t["sentence_num"] - 1][-1])

    event_count = 2
    event_class_count = 2

    if predict_event is True:

        if predicate_as_event is False:

            eventClassifier = _models["EVENT"]
            eventVectorizer = _vects["EVENT"]

            # get the timex feature set for the tokens within the note.
            # don't get iob labels yet, they are inaccurate. need to predict first.
            eventFeatures = features.extract_event_feature_set(note, eventLabels, predict=True, timexLabels=timexLabels)

            # sanity check
            assert len(tokens) == len(eventFeatures)

            # TODO: need to do some filter. if something is already labeled then just skip over it.
            # predict over the tokens and the features extracted.
            for t, f in zip(tokens, eventFeatures):

                features.update_features(t, f, eventLabels)

                X = eventVectorizer.transform([f])
                Y = list(eventClassifier.predict(X))

                eventLabels[t["sentence_num"] - 1].append({'entity_label':Y[0],
                                                           'entity_type':None if Y[0] == 'O' else 'EVENT',
                                                           'entity_id':"e" + str(event_count)})

                event_count += 1
        else:
            for t in tokens:
                if t["is_predicate"]:
                    eventLabels[t["sentence_num"] - 1].append({'entity_label':'EVENT',
                                                               'entity_type':'EVENT',
                                                               'entity_id':"t"+str(event_count)})
                else:
                    eventLabels[t["sentence_num"] - 1].append({'entity_label':'O',
                                                               'entity_type':None,
                                                               'entity_id':"t"+str(event_count)})
                event_count += 1

        #print
        #print "eventLabels is None: ", eventLabels is None
        #print

        eventClassClassifier = _models["EVENT_CLASS"]
        eventClassVectorizer = _vects["EVENT_CLASS"]

        # get the timex feature set for the tokens within the note.
        eventClassFeatures = features.extract_event_class_feature_set(note, eventClassLabels, eventLabels, predict=True, timexLabels=timexLabels)

        # sanity check
        assert len(tokens) == len(eventClassFeatures)

        # predict over the tokens and the features extracted.
        for t, f in zip(tokens, eventClassFeatures):

            X = None
            Y = None

            # updates labels
            features.update_features(t, f, eventClassLabels)

            if predicate_as_event:
                if t["is_predicate"]:
                    X = eventClassVectorizer.transform([f])
                    Y = list(eventClassClassifier.predict(X))[0]
                else:
                    Y = 'O'
            else:
                X = eventClassVectorizer.transform([f])
                Y = list(eventClassClassifier.predict(X))[0]

            eventClassLabels[t["sentence_num"] - 1].append({'entity_label':Y,
                                                            'entity_type':None if Y == 'O' else 'EVENT',
                                                            'entity_id':'e' + str(event_class_count)})

            event_class_count += 1

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

        f = features.extract_tlink_features(note, event_labels=eventClassLabels, timex_labels=timexLabels)
        X = tlinkVectorizer.transform(f)

        tlink_labels = list(tlinkClassifier.predict(X))

    entity_labels    = [label for line in iob_labels for label in line]
    original_offsets = note.get_token_char_offsets()

    # print entity_labels

    return entity_labels, original_offsets, tlink_labels, tokens


def _trainEvent(eventFeatures, eventLabels, grid=False):
    """
    Model::_trainEvent()

    Purpose: Train a classifer for event identification

    @param tokenVectors: A list of tokens represented as feature dictionaries
    @param Y: A list of lists of event classifications for each token, with one list per sentence
    """

    assert len(eventFeatures) == len(eventLabels), "{} != {}".format(len(eventFeatures), len(eventLabels))

    Y = [l["entity_label"] for l in eventLabels]

    sys.exit("dev: never call this...")

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

    clf, vec = train_classifier(eventClassFeatures, Y, do_grid=grid, t="EVENT_CLASS")
    return clf, vec


def _trainTlink(tokenVectors, Y, grid=False):
    """
    Model::_trainRelation()

    Purpose: Train a classifer for temporal relations between events and timex3 labels

    @param tokenVectors: A list of tokens represented as feature dictionaries
    @param Y: A list of relation classifications for each pair of timexes and events.
    """

    print len(tokenVectors)
    print len(Y)

    assert len(tokenVectors) == len(Y)

    clf, vec = train_classifier(tokenVectors, Y, do_grid=grid, ovo=True, t="TLINK")
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

def load_models(path, predict_event, predict_tlink, predicate_as_event):

    keys = ["EVENT", "EVENT_CLASS", "TLINK"]
    flags = [predict_event, predict_event, predict_tlink]

    global _models
    global _vects
    global _models_loaded

    print path

    for key, flag in zip(keys, flags):

        m_path = path+"_"+key+"_MODEL"
        v_path = path+"_"+key+"_VECT"

        print
        print "key: ", key
        print "flag: ", flag
        print "m_path: ", m_path
        print "v_path: ", v_path
        print


        if predicate_as_event and key == "EVENT_CLASS":
            m_path += "_PREDICATE_AS_EVENT"
            v_path += "_PREDICATE_AS_EVENT"

        # vect should also exist, unless something went wrong.
        if os.path.isfile(m_path) is True and flag is True:
            print "SETTING: ", key
            print "loading: {}".format(key)
            _models[key] = cPickle.load(open(m_path, "rb"))
            _vects[key]  = cPickle.load(open(v_path, "rb"))
        else:
            _models[key] = None
            _vects[key]  = None

    _models_loaded = True

    print _vects
    print _models

    return


def dump_models(models, vectorizers, path, predicate_as_event):
    """dump model specified by argument into the file path indicated by path argument
    """

    print "dumping..."

    keys = ["EVENT", "EVENT_CLASS", "TLINK"]

    for key in keys:
        if models[key] is None:
            continue
        else:
            print "dumping: {}".format(key)

            model_dest = None
            vect_dest  = None

            if predicate_as_event and key == "EVENT_CLASS":
                model_dest = open(path+"_"+key+"_MODEL"+"_PREDICATE_AS_EVENT", "wb")
                vect_dest  = open(path+"_"+key+"_VECT"+"_PREDICATE_AS_EVENT", "wb")
            elif key =="EVENT_CLASS":
                model_dest = open(path+"_"+key+"_MODEL"+"_REGULAR_EVENT", "wb")
                vect_dest  = open(path+"_"+key+"_VECT"+"_REGULAR_EVENT", "wb")
            else:
                model_dest = open(path+"_"+key+"_MODEL", "wb")
                vect_dest  = open(path+"_"+key+"_VECT", "wb")


            cPickle.dump(models[key], model_dest)
            cPickle.dump(vectorizers[key], vect_dest)

    return

