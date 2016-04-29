import os

from code.notes import features
from code.notes.TimeNote import TimeNote
from machine_learning.sci import train as train_classifier

class Model:


    def __init__(self, grid=False):

        self.grid=grid

    def train(self, notes):

        # TODO: need to do some filtering of tokens
        # TODO: experiment with the feature of the 4 left and right taggings. do we
        #       only utilize taggings for each pass or do we incorporate taggings in different passes?

        timexLabels   = []
        timexFeatures = []
        tlinkLabels   = []

        eventLabels   = []
        eventFeatures = []

        eventClassLabels   = []
        eventClassFeatures = []

        tlinkFeatures = []

        for i, note in enumerate(notes):

            print "note: {}".format(i)

            # get timex labels
            tmpLabels = note.get_timex_labels()

            for label in tmpLabels:
                timexLabels += label

            timexFeatures += features.extract_timex_feature_set(note, tmpLabels)

            tmpLabels = note.get_event_labels()

            for label in tmpLabels:
                eventLabels += label

            eventFeatures += features.extract_event_feature_set(note, tmpLabels)

            tmpLabels = note.get_event_class_labels()

            for label in tmpLabels:
                eventClassLabels += label

            eventClassFeatures += features.extract_event_class_feature_set(note, tmpLabels, note.get_event_labels())

            tlinkLabels = note.get_tlink_labels()

            print "tlinkLabels: ", tlinkLabels

            tlinkFeatures += features.extract_tlink_features(note)

        self._trainTimex(timexFeatures, timexLabels)

        # TODO: filter non-timex only?
        self._trainEvent(eventFeatures, eventLabels)

        # TODO: filter event only?
        # TODO: should we be training over all tokens or those that are just EVENTs?
        self._trainEventClass(eventClassFeatures, eventClassLabels)

        # TODO: add features back in.
        self._trainTlink(tlinkFeatures, tlinkLabels)

        return




    def predict(self, note):

        # TODO: try and correct the flattening on the lists. might just end up being redundent?
        # TODO: refactor this code. a lot of it is redundant.
        # TODO: need to do some filtering of tokens
        # TODO: experiment with the feature of the 4 left and right taggings. do we
        #       only utilize taggings for each pass or do we incorporate taggings in different passes?

        # get tokenized text
        tokenized_text = note.get_tokenized_text()

        # will be new iob_labels
        iob_labels      = []

        timexLabels      = []
        eventLabels      = []
        eventClassLabels = []

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

        # get the timex feature set for the tokens within the note.
        timexFeatures = features.extract_timex_feature_set(note, timexLabels, predict=True)

        # sanity check
        assert len(tokens) == len(timexFeatures)

        # predict over the tokens and the features extracted.
        for t, f in zip(tokens, timexFeatures):

            features.update_features(t, f, timexLabels)

            X = self.timexVectorizer.transform([f]).toarray()
            Y = list(self.timexClassifier.predict(X))

            timexLabels[t["sentence_num"] - 1].append({'entity_label':Y[0],
                                                       'entity_type':None if Y[0] == 'O' else 'TIMEX3',
                                                       'entity_id':None})

            iob_labels[t["sentence_num"] - 1].append(timexLabels[t["sentence_num"] - 1][-1])

        # get the timex feature set for the tokens within the note.
        # don't get iob labels yet, they are inaccurate. need to predict first.
        eventFeatures = features.extract_event_feature_set(note, eventLabels, predict=True)

        # sanity check
        assert len(tokens) == len(eventFeatures)

        # TODO: need to do some filter. if something is already labeled then just skip over it.
        # predict over the tokens and the features extracted.
        for t, f in zip(tokens, eventFeatures):

            features.update_features(t, f, eventLabels)

            X = self.eventVectorizer.transform([f]).toarray()
            Y = list(self.eventClassifier.predict(X))

            eventLabels[t["sentence_num"] - 1].append({'entity_label':Y[0],
                                                       'entity_type':None if Y[0] == 'O' else 'EVENT',
                                                       'entity_id':None})

        # get the timex feature set for the tokens within the note.
        eventClassFeatures = features.extract_event_class_feature_set(note, eventClassLabels, eventLabels, predict=True)

        # sanity check
        assert len(tokens) == len(eventClassFeatures)

        i = 0
        sentence_num = None

        # predict over the tokens and the features extracted.
        for t, f in zip(tokens, eventClassFeatures):

            # updates labels
            features.update_features(t, f, eventClassLabels)

            X = self.eventClassVectorizer.transform([f]).toarray()
            Y = list(self.eventClassClassifier.predict(X))

            eventClassLabels[t["sentence_num"] - 1].append({'entity_label':Y[0],
                                                            'entity_type':None if Y[0] == 'O' else 'EVENT',
                                                            'entity_id':None})

            if sentence_num is None:
                sentence_num = t["sentence_num"] - 1
            # new sentence
            elif sentence_num != t["sentence_num"] - 1:
                sentence_num = t["sentence_num"] - 1
                i = 0
            else:
                pass

            if iob_labels[t["sentence_num"] - 1][i]["entity_type"] == None:
                iob_labels[t["sentence_num"] - 1][i] = eventClassLabels[t["sentence_num"] - 1][-1]

            i += 1

        note.set_tlinked_entities(timexLabels,eventClassLabels)
        note.set_iob_labels(iob_labels)

        print "PREDICT: getting tlink features"

        print features.extract_tlink_features(note)

        exit()

        return



    def _trainTimex(self, timexFeatures, timexLabels):
        """
        Model::_trainTimex()

        Purpose: Train a classifer for Timex3 expressions

        @param tokenVectors: A list of tokens represented as feature dictionaries
        @param Y: A list of lists of Timex3 classifications for each token in each sentence
        """

        assert len(timexFeatures) == len(timexLabels), "{} != {}".format(len(timexFeatures), len(timexLabels))

        Y = [l["entity_label"] for l in timexLabels]

        clf, vec = train_classifier(timexFeatures, Y, do_grid=self.grid, ovo=True, dev=True)
        self.timexClassifier = clf
        self.timexVectorizer = vec


    def _trainEvent(self, eventFeatures, eventLabels):
        """
        Model::_trainEvent()

        Purpose: Train a classifer for event identification

        @param tokenVectors: A list of tokens represented as feature dictionaries
        @param Y: A list of lists of event classifications for each token, with one list per sentence
        """

        assert len(eventFeatures) == len(eventLabels), "{} != {}".format(len(eventFeatures), len(eventLabels))

        Y = [l["entity_label"] for l in eventLabels]

        clf, vec = train_classifier(eventFeatures, Y, do_grid=self.grid, dev=True)
        self.eventClassifier = clf
        self.eventVectorizer = vec

    def _trainEventClass(self, eventClassFeatures, eventClassLabels):
        """
        Model::_trainEventClass()

        Purpose: Train a classifer for event class identification

        @param tokenVectors: A list of tokens represented as feature dictionaries
        @param Y: A list of lists of event classifications for each token, with one list per sentence
        """

        assert len(eventClassFeatures) == len(eventClassLabels), "{} != {}".format(len(eventClassFeatures), len(eventClassLabels))

        Y = [l["entity_label"] for l in eventClassLabels]

        clf, vec = train_classifier(eventClassFeatures, Y, do_grid=self.grid, dev=True)
        self.eventClassClassifier = clf
        self.eventClassVectorizer = vec

    def _trainTlink(self, tokenVectors, Y):
        """
        Model::_trainRelation()

        Purpose: Train a classifer for temporal relations between events and timex3 labels

        @param tokenVectors: A list of tokens represented as feature dictionaries
        @param Y: A list of relation classifications for each pair of timexes and events.
        """

        assert len(tokenVectors) == len(Y)

        clf, vec = train_classifier(tokenVectors, Y, do_grid=self.grid, dev=True)
        self.tlinkClassifier = clf
        self.tlinkVectorizer = vec

def combineLabels(timexLabels, eventLabels, OLabels=[]):
    """
    combineTimexEventLabels():
        merge event and timex labels into one list, adding instance ids

    @param timexLabels: list of timex labels for entities.
    @param eventLabels: list of event labels for entities. Includes no instances labeled as timexs
    @return: list of dictionaries, with one dictionary for each entity
    """

    labels = []

    creation time is always t0
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

