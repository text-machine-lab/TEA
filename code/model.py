import os

from code.notes.TimeNote import TimeNote
from machine_learning.sci import train as train_classifier

class Model:

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
			eventTimexFeats = eventTimexFeats + note.get_iob_features()

			#timex labels 
			tmpLabels = note.get_timex_iob_labels()
			for label in tmpLabels:
				timexLabels = timexLabels + label

			#event labels
			tmpLabels = note.get_event_iob_labels()
			for label in tmpLabels:
				eventLabels = eventLabels + label

			#tlink features and labels
			tlinkFeats = tlinkFeats + note.get_tlink_features()
			tlinkLabels = tlinkLabels + note.get_tlink_labels()
		
			#delete these after testing
			tlinkIds = note.get_tlink_id_pairs()
			offsets = note.get_token_char_offsets()
			tLabels = []
			eLabels = []
			for label in timexLabels:
				tLabels.append(label["entity_label"])
			for label in eventLabels:
				eLabels.append(label["entity_label"]) 

			# note.write(tLabels, eLabels, tlinkLabels, tlinkIds, offsets)

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
		eventTimexFeats	= note.get_iob_features()

		#TODO: move direct SVM interfacing back to sci.py

		#vectorize
		timexVec = self.timexVectorizer.transform(eventTimexFeats).toarray()
		eventVec = self.eventVectorizer.transform(eventTimexFeats).toarray()

		#classify
		timexLabels = list(self.timexClassifier.predict(timexVec))
		eventLabels = list(self.eventClassifier.predict(eventVec))

		#write timex and events to annoation file
		note.write(timexLabels, eventLabels, None, None, offsets)
		#set new annoation path for the note so tlink data generates properly
		note._set_note_path(note.note_path, (os.environ['TEA_PATH'] + '/output/' + note.note_path.split('/')[-1][:-9]))

		# note.get_tlinked_entities()

		tlinkFeats = note.get_tlink_features()

		tlinkVec = self.tlinkVectorizer.transform(tlinkFeats).toarray()
		tlinkLabels = list(self.tlinkClassifier.predict(tlinkVec))

		print tlinkLabels

		return timexLabels, eventLabels, offsets

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

		clf, vec = train_classifier(tokenVectors, Y)
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

		clf, vec = train_classifier(tokenVectors, Y)
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
		
		clf, vec = train_classifier(tokenVectors, Y)
		self.tlinkClassifier = clf
		self.tlinkVectorizer = vec
	

