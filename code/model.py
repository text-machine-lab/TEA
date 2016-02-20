import os

from code.notes.TimeNote import TimeNote
from utilities import combineLabels
from machine_learning.sci import train as train_classifier


class Model:


    def __init__(self, grid=False):

        self.grid=grid


    def train(self, notes):
        """
        Model::train()

        Purpose: Train a classification model for Timex3, events, and temporal event relations from annotated data

        @param notes. A list of TimeNote objects containing annotated data
        """

        timexLabels = []
        eventLabels = []
        tlinkFeats  = []
        tlinkLabels = []

        tlinkIds = []
        offsets  = []

        eventFeats = []
        timexFeats = []

        #populate feature and label lists
        for note in notes:

            eventFeats += note.get_event_features()
            timexFeats += note.get_timex_features()

            #timex labels
            tmpLabels = note.get_timex_iob_labels()
            for label in tmpLabels:
                timexLabels += label

            #event labels
            tmpLabels = note.get_event_iob_labels()
            for label in tmpLabels:
                eventLabels += label

            #tlink features and labels
            tlinkFeats += note.get_tlink_features()
            tlinkLabels += note.get_tlink_labels()

        assert len(eventFeats) == len(eventLabels), "{} != {}".format(len(eventFeats), len(eventLabels))
        assert len(timexFeats) == len(timexLabels), "{} != {}".format(len(timexFeats), len(timexLabels))
        assert len(tlinkFeats) == len(tlinkLabels), "{} != {}".format(len(tlinkFeats), len(tlinkLabels))

        #train classifiers
        self._trainTimex(timexFeats, timexLabels)
        self._trainEvent(eventFeats, eventLabels)

        self._trainTlink(tlinkFeats, tlinkLabels)


    def predict(self, note):
        """
        Model::predict()

        Purpose: Predict temporal relations for a set of notes

        @param notes: A list of TimeNote objects
        @return: All three label sets predicted by the classifiers
        """

        timexLabels = []
        eventLabels = []
        tlinkFeats  = []
        tlinkLabels = []
        offsets     = note.get_token_char_offsets()

        #populate feature lists
        _timexFeats = note.get_timex_features()

        tokens  = note.get_tokens()

        # reconstruct labelings
        iob_labels = []

        tokenized_text = note.get_tokenized_text()

        assert len(tokens) == len(_timexFeats)
        assert len(offsets) == len(_timexFeats)

        #TODO: move direct SVM interfacing back to sci.py

        #vectorize and classify timex
        potentialTimexVec = self.timexVectorizer.transform(_timexFeats).toarray()
        timexLabels_withO = list(self.timexClassifier.predict(potentialTimexVec))

        # adjust labels. normalize tool doesn't like this.
        for i, token in enumerate(tokens):
            if len(token["token"]) == 1:
                timexLabels_withO[i] = 'O'

        #filter out all timex-labeled entities
        timexFeats = []
        timexOffsets = []
        timexLabels = []
        potentialEventOffsets = []
        potentialEventFeats = []

        potentialEventTokens = []
        timexTokens = []

        _tmp_labels = []

        for line in tokenized_text:
            _tmp_labels.append([ {'entity_label': 'O', 'entity_type': None}] * len(tokenized_text[line]))

        for token, label in zip(tokens, timexLabels_withO):

            _tmp_labels[token["sentence_num"] - 1][token["token_offset"]]["entity_label"]=label
            _tmp_labels[token["sentence_num"] - 1][token["token_offset"]]["entity_type"]=label[2:]

        note.set_iob_labels(_tmp_labels)

        _eventFeats = note.get_event_features()

        for i in range(0, len(timexLabels_withO)):
            if timexLabels_withO[i] == 'O':
                potentialEventFeats.append(_eventFeats[i])
                potentialEventOffsets.append(offsets[i])

                potentialEventTokens.append(tokens[i])

            else:
                timexFeats.append(_timexFeats[i])
                timexOffsets.append(offsets[i])
                timexLabels.append(timexLabels_withO[i])

                timexTokens.append(tokens[i])

        assert len(potentialEventFeats + timexFeats) == len(offsets), "{} != {}".format(len(offsets), + len(potentialEventFeats + timexFeats))
        assert len(potentialEventTokens + timexTokens) == len(offsets)

        #vectorize and classify events
        potentialEventVec = self.eventVectorizer.transform(potentialEventFeats).toarray()
        eventLabels_withO = list(self.eventClassifier.predict(potentialEventVec))

        # adjust labels. normalize tool doesn't like this.
        for i, token in enumerate(potentialEventTokens):
            if len(token["token"]) == 1:
                eventLabels_withO[i] = 'O'

        OOffsets = []
        OFeats = []
        OLabels = []
        eventOffsets = []
        eventFeats = []
        eventLabels = []

        OTokens = []
        eventTokens = []

        for i in range(0, len(eventLabels_withO)):
            if eventLabels_withO[i] == 'O':
                OOffsets.append(potentialEventOffsets[i])
                OFeats.append(potentialEventFeats[i])
                OLabels.append(eventLabels_withO[i])
                OTokens.append(potentialEventTokens[i])

        else:
            eventFeats.append(potentialEventFeats[i])
            eventOffsets.append(potentialEventOffsets[i])
            eventLabels.append(eventLabels_withO[i])

            eventTokens.append(potentialEventTokens[i])

        assert len(OOffsets + eventOffsets + timexOffsets) == len(eventFeats + OFeats + timexFeats)
        assert len(OLabels + eventLabels + timexLabels) == len(eventFeats + OFeats + timexFeats)
        assert len(offsets) == len(OOffsets + eventOffsets + timexOffsets),
               "len(offsets): {}, len(OOffsets + eventOffsets + timexOffsets): {}".format(len(offsets),
                                                                                          len(OOffsets + eventOffsets + timexOffsets))
        assert len(OLabels + eventLabels + timexLabels) == len(OTokens + eventTokens + timexTokens)

        timexEventLabels = combineLabels(timexLabels, eventLabels, OLabels)
        timexEventFeats = timexFeats + eventFeats + OFeats
        timexEventOffsets = timexOffsets + eventOffsets + OOffsets
        timexEventTokens = timexTokens + eventTokens + OTokens

        assert len(timexEventOffsets) == len(timexEventTokens)
        assert len(timexEventOffsets) == len(timexEventFeats)
        assert len(timexEventLabels) == len(timexEventFeats)

        offsetDictLabel = {}
        offsetDictToken = {}

        for i, offset in enumerate(timexEventOffsets):
            offsetDictToken[offset] = timexEventTokens[i]
            offsetDictLabel[offset] = timexEventLabels[i]

        wellOrderedEntityLabels = []
        wellOrderedTokens = []

        for offset in note.get_token_char_offsets():
            wellOrderedEntityLabels.append(offsetDictLabel[offset])
            wellOrderedTokens.append(offsetDictToken[offset])

        assert len(offsetDictToken) == len(offsetDictLabel)

        unflattened_iobs = []

        # TODO: awful code. last minute bug fixing :(
        for line in tokenized_text:
            unflattened_iobs.append([ {'entity_label': 'O', 'entity_id': None, 'entity_type': None}] * len(tokenized_text[line]))

        for tokenKey, labelKey in zip(offsetDictToken, offsetDictLabel):

            assert tokenKey == labelKey
            unflattened_iobs[offsetDictToken[tokenKey]["sentence_num"] - 1][offsetDictToken[tokenKey]["token_offset"]] = offsetDictLabel[labelKey]

        note.set_iob_labels(unflattened_iobs)

        note.set_tlinks(wellOrderedTokens, wellOrderedEntityLabels)

        tlinkFeats = note.get_tlink_features()

        tlinkVec = self.tlinkVectorizer.transform(tlinkFeats).toarray()

        tlinkLabels = list(self.tlinkClassifier.predict(tlinkVec))

        return wellOrderedEntityLabels, timexEventOffsets, tlinkLabels, wellOrderedTokens


    def _trainTimex(self, tokenVectors, labels):
        """
        Model::_trainTimex()

        Purpose: Train a classifer for Timex3 expressions

        @param tokenVectors: A list of tokens represented as feature dictionaries
        @param Y: A list of lists of Timex3 classifications for each token in each sentence
        """

        assert len(tokenVectors) == len(labels)

        Y = []

        for label in labels:
            Y.append(label["entity_label"])

        clf, vec = train_classifier(tokenVectors, Y, do_grid=self.grid)
        self.timexClassifier = clf
        self.timexVectorizer = vec


    def _trainEvent(self, tokenVectors, labels):
        """
        Model::_trainEvent()

        Purpose: Train a classifer for event identification

        @param tokenVectors: A list of tokens represented as feature dictionaries
        @param Y: A list of lists of event classifications for each token, with one list per sentence
        """

        assert len(tokenVectors) == len(labels)

        Y = []


        for label in labels:
            Y.append(label["entity_label"])

        clf, vec = train_classifier(tokenVectors, Y, do_grid=self.grid)
        self.eventClassifier = clf
        self.eventVectorizer = vec


    def _trainTlink(self, tokenVectors, Y):
        """
        Model::_trainRelation()

        Purpose: Train a classifer for temporal relations between events and timex3 labels

        @param tokenVectors: A list of tokens represented as feature dictionaries
        @param Y: A list of relation classifications for each pair of timexes and events.
        """

        assert len(tokenVectors) == len(Y)

        clf, vec = train_classifier(tokenVectors, Y, do_grid=self.grid)
        self.tlinkClassifier = clf
        self.tlinkVectorizer = vec


