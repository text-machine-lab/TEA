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
		for i in range(0, len(timexLabels_withO)):
			if timexLabels_withO[i] == 'O':
				potentialEventFeats.append(timexEventFeats[i])
				potentialEventOffsets.append(offsets[i])
			else:
				timexFeats.append(timexEventFeats[i])
				timexOffsets.append(offsets[i])
				timexLabels.append(timexLabels_withO[i])

		assert len(potentialEventFeats + timexFeats) == len(offsets), "{} != {}".format(len(offsets), + len(potentialEventFeats + timexFeats))

		#vectorize and classify events
		potentialEventVec = self.eventVectorizer.transform(potentialEventFeats).toarray()
		eventLabels_withO = list(self.eventClassifier.predict(potentialEventVec))

		OOffsets = []
		OFeats = []
		OLabels = []
		eventOffsets = []
		eventFeats = []
		eventLabels = []
		for i in range(0, len(eventLabels_withO)):
			if eventLabels_withO[i] == 'O':
				OOffsets.append(potentialEventOffsets[i])
				OFeats.append(potentialEventFeats[i])
				OLabels.append(eventLabels_withO[i])
			else:
				eventFeats.append(potentialEventFeats[i])
				eventOffsets.append(potentialEventOffsets[i])
				eventLabels.append(eventLabels_withO[i])

		assert len(OOffsets + eventOffsets + timexOffsets) == len(eventFeats + OFeats + timexFeats)
		assert len(OLabels + eventLabels + timexLabels) == len(eventFeats + OFeats + timexFeats)
		assert len(offsets) == len(OOffsets + eventOffsets + timexOffsets), "len(offsets): {}, len(OOffsets + eventOffsets + timexOffsets): {}".format(len(offsets), len(OOffsets + eventOffsets + timexOffsets))

		timexEventLabels = combineLabels(timexLabels, eventLabels, OLabels)

		timexEventFeats = timexFeats + eventFeats + OFeats
		timexEventOffsets = timexOffsets + eventOffsets + OOffsets

		assert len(timexEventOffsets) == len(timexEventFeats)
		assert len(timexEventLabels) == len(timexEventFeats)

		note.set_tlinks(timexEventFeats, timexEventLabels, timexEventOffsets)

		tlinkFeats = note.get_tlink_features()

		tlinkVec = self.tlinkVectorizer.transform(tlinkFeats).toarray()
		tlinkLabels = list(self.tlinkClassifier.predict(tlinkVec))

		print tlinkLabels

		return timexEventLabels, timexEventOffsets

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


