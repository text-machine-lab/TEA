from code.notes.TimeNote import TimeNote

from machine_learning.sci import train as train_classifier

class Model:

	def train(self, notes):
		'''
		Model::train()

		Purpose: Train a classification model for Timex3, events, and temporal event relations from annotated data

		@param notes. A list of TimeNote objects containing annotated data
		'''

		timexFeats		= []
		timexLabels		= []
		eventFeats		= []
		eventLabels		= []
		relationFeats	= []
		relationLabels	= []

		#populate feature and label lists
		for note in notes:
			
			tmpFeats, tmpLabels = note.vectorize("TIMEX3")
			timexFeats = timexFeats + tmpFeats
			timexLabels = timexLabels + tmpLabels

			tmpFeats, tmpLabels = note.vectorize("EVENT")
			eventFeats = eventFeats + tmpFeats
			eventLabels = eventLabels + tmpLabels


		# print timexFeats
		#train classifiers
		self._trainTimex(timexFeats, timexLabels)
		self._trainEvent(eventFeats, eventLabels)
		# self._trainRelation(relationFeats, relationLabels)

	def predict(self, note):
		'''
		Model::predict()

		Purpose: Predict temporal relations for a set of notes

		@param notes: A list of TimeNote objects
		@return: All three label sets predicted by the classifiers
		'''

		timexFeats		= []
		timexLabels		= []
		eventFeats		= []
		eventLabels		= []
		relationFeats	= []
		relationLabels	= []
		offsets			= []

		#populate feature lists
		tmpFeats, tmpLabels, offsets = note.vectorize("TIMEX3")
		timexFeats = timexFeats + tmpFeats

		tmpFeats, tmpLabels, offsets = note.vectorize("EVENT")
		eventFeats = eventFeats + tmpFeats

		#TODO: move direct SVM interfacing back to sci.py

		#vectorize
		timexVec = self.timexVectorizer.transform(timexFeats).toarray()
		eventVec = self.eventVectorizer.transform(eventFeats).toarray()

		#classify
		timexLabels = list(self.timexClassifier.predict(timexVec))
		eventLabels = list(self.eventClassifier.predict(eventVec))

		return timexLabels, eventLabels, offsets

	def _trainTimex(self, tokenVectors, Y):
		'''
		Model::_trainTimex()

		Purpose: Train a classifer for Timex3 expressions

		@param tokenVectors: A list of tokens represented as feature dictionaries
		@param Y: A list of lists of Timex3 classifications for each token in each sentence
		'''

		clf, vec = train_classifier(tokenVectors, Y)
		self.timexClassifier = clf
		self.timexVectorizer = vec


	def _trainEvent(self, tokenVectors, Y):
		'''
		Model::_trainEvent()

		Purpose: Train a classifer for event identification

		@param tokenVectors: A list of tokens represented as feature dictionaries
		@param Y: A list of lists of event classifications for each token, with one list per sentence
		'''

		clf, vec = train_classifier(tokenVectors, Y)
		self.eventClassifier = clf
		self.eventVectorizer = vec


	def _trainRelation(self, tokenVectors, Y):
		'''
		Model::_trainRelation()

		Purpose: Train a classifer for temporal relations between events and timex3 labels

		@param tokenVectors: A list of tokens represented as feature dictionaries
		@param Y: A list of relation classifications for each pair of timexes and events.
		'''
		
		clf, vec = train_classifier(tokenVectors, Y)
		self.tlinkClassifier = clf
		self.tlinkVectorizer = vec
	

