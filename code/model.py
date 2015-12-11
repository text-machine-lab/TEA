import os

from code.notes.TimeNote import TimeNote
from utilities import combineLabels
from machine_learning.sci import train as train_classifier

class Model:

	def __init__(self, grid=False):

		self.grid=grid

	def train(self, notes):
		'''
		Model::train()

		Purpose: Train a classification model for Timex3, events, and temporal event relations from annotated data

		@param notes. A list of TimeNote objects containing annotated data
		'''

		eventTimexFeats	= []
		timexLabels	= []
		eventLabels	= []
		tlinkFeats	= []
		tlinkLabels	= []

		tlinkIds	= []
		offsets 	= []


		#populate feature and label lists
		for note in notes:

			#timex and event features
			eventTimexFeats += note.get_iob_features()

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

		#train classifiers
		self._trainTimex(eventTimexFeats, timexLabels)
		self._trainEvent(eventTimexFeats, eventLabels)

		self._trainTlink(tlinkFeats, tlinkLabels)

	def predict(self, note):
		'''
		Model::predict()

		Purpose: Predict temporal relations for a set of notes

		@param notes: A list of TimeNote objects
		@return: All three label sets predicted by the classifiers
		'''

		timexLabels	= []
		eventLabels	= []
		tlinkFeats	= []
		tlinkLabels	= []
		offsets		= note.get_token_char_offsets()

		#populate feature lists
		timexEventFeats = note.get_iob_features()

		tokens          = note.get_tokens()

		# reconstruct labelings
		iob_labels = []

		tokenized_text = note.get_tokenized_text()

		for line in tokenized_text:
			iob_labels.append([None] * len(tokenized_text[line]))

		assert len(tokens) == len(timexEventFeats)
		assert len(offsets) == len(timexEventFeats)

		#TODO: move direct SVM interfacing back to sci.py

		#vectorize and classify timex
		potentialTimexVec = self.timexVectorizer.transform(timexEventFeats).toarray()
		timexLabels_withO = list(self.timexClassifier.predict(potentialTimexVec))

		#filter out all timex-labeled entities
		timexFeats = []
		timexOffsets = []
		timexLabels = []
		potentialEventOffsets = []
		potentialEventFeats = []

		potentialEventTokens = []
		timexTokens = []

		for i in range(0, len(timexLabels_withO)):
			if timexLabels_withO[i] == 'O':
				potentialEventFeats.append(timexEventFeats[i])
				potentialEventOffsets.append(offsets[i])

				potentialEventTokens.append(tokens[i])

			else:
				timexFeats.append(timexEventFeats[i])
				timexOffsets.append(offsets[i])
				timexLabels.append(timexLabels_withO[i])

				timexTokens.append(tokens[i])

		assert len(potentialEventFeats + timexFeats) == len(offsets), "{} != {}".format(len(offsets), + len(potentialEventFeats + timexFeats))
		assert len(potentialEventTokens + timexTokens) == len(offsets)

		#vectorize and classify events
		potentialEventVec = self.eventVectorizer.transform(potentialEventFeats).toarray()
		eventLabels_withO = list(self.eventClassifier.predict(potentialEventVec))

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
				OTokens.append(tokens[i])

			else:
				eventFeats.append(potentialEventFeats[i])
				eventOffsets.append(potentialEventOffsets[i])
				eventLabels.append(eventLabels_withO[i])

				eventTokens.append(tokens[i])

		assert len(OOffsets + eventOffsets + timexOffsets) == len(eventFeats + OFeats + timexFeats)
		assert len(OLabels + eventLabels + timexLabels) == len(eventFeats + OFeats + timexFeats)
		assert len(offsets) == len(OOffsets + eventOffsets + timexOffsets), "len(offsets): {}, len(OOffsets + eventOffsets + timexOffsets): {}".format(len(offsets), len(OOffsets + eventOffsets + timexOffsets))
		assert len(OLabels + eventLabels + timexLabels) == len(OTokens + eventTokens + timexTokens)

		timexEventLabels = combineLabels(timexLabels, eventLabels)
		timexEventFeats = timexFeats + eventFeats
		timexEventOffsets = timexOffsets + eventOffsets
		timexEventTokens = timexTokens + eventTokens

		assert len(timexEventOffsets) == len(timexEventTokens)
		assert len(timexEventOffsets) == len(timexEventFeats)
		assert len(timexEventLabels) == len(timexEventFeats)

		for token, label in zip(timexEventTokens, timexEventLabels):
			sentence_index = token["sentence_num"] - 1
			token_index = token["token_offset"]
			iob_labels[sentence_index][token_index] = label

		note.set_iob_labels(iob_labels)

		assert iob_labels == note.get_iob_labels()

		note.set_tlinks(timexEventTokens, timexEventLabels, timexEventOffsets)

		tlinkFeats = note.get_tlink_features()

		tlinkVec = self.tlinkVectorizer.transform(tlinkFeats).toarray()

		tlinkLabels = list(self.tlinkClassifier.predict(tlinkVec))

		timexEventOffsets = timexOffsets + eventOffsets + OOffsets

		timexEventLabels  = combineLabels(timexLabels, eventLabels, OLabels)

		return timexEventLabels, timexEventOffsets, tlinkLabels

	def _trainTimex(self, tokenVectors, labels):
		'''
		Model::_trainTimex()

		Purpose: Train a classifer for Timex3 expressions

		@param tokenVectors: A list of tokens represented as feature dictionaries
		@param Y: A list of lists of Timex3 classifications for each token in each sentence
		'''

		assert len(tokenVectors) == len(labels)

		Y = []

		for label in labels:
			Y.append(label["entity_label"])

		clf, vec = train_classifier(tokenVectors, Y, do_grid=self.grid)
		self.timexClassifier = clf
		self.timexVectorizer = vec


	def _trainEvent(self, tokenVectors, labels):
		'''
		Model::_trainEvent()

		Purpose: Train a classifer for event identification

		@param tokenVectors: A list of tokens represented as feature dictionaries
		@param Y: A list of lists of event classifications for each token, with one list per sentence
		'''

		assert len(tokenVectors) == len(labels)

		Y = []


		for label in labels:
			Y.append(label["entity_label"])

		clf, vec = train_classifier(tokenVectors, Y, do_grid=self.grid)
		self.eventClassifier = clf
		self.eventVectorizer = vec


	def _trainTlink(self, tokenVectors, Y):
		'''
		Model::_trainRelation()

		Purpose: Train a classifer for temporal relations between events and timex3 labels

		@param tokenVectors: A list of tokens represented as feature dictionaries
		@param Y: A list of relation classifications for each pair of timexes and events.
		'''

		assert len(tokenVectors) == len(Y)

		clf, vec = train_classifier(tokenVectors, Y, do_grid=self.grid)
		self.tlinkClassifier = clf
		self.tlinkVectorizer = vec


