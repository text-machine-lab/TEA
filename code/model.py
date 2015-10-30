from code.notes.TimeNote import TimeNote

from machine_learning.sci import train as train_classifier

class Model:

	def train(self, notes):
		'''
		Model::train()

		Purpose: Train a classification model for Timex3, events, and temporal event relations from annotated data

		@param notes. A list of TimeNote objects containing annotated data
		'''

		#TODO: add extraction methods that actually get features from annotated data
		
		tokenVectors = []
		timexLabels = []
		eventLabels = []
		relationLabels = []

		#create flat list of token vectors
		for note in notes:
			tmpVecs = note.vectorize()
			for sent in tmpVecs:
				for vec in sent:
					tokenVectors.append(vec)
					if(vec["word"] == 'a'):
						timexLabels.append("NONE")
					elif(vec["word"] == 'the'):
						timexLabels.append("CATS")
					else:
						timexLabels.append("TEST")


		# print tokenVectors
		#train classifiers
		self._trainTimex(tokenVectors, timexLabels)
		# self._trainEvent(tokenVectors, eventLabels)
		# self._trainRelation(tokenVectors, timexLabels, eventLabels, relationLabels)

	def predict(self, notes):
		'''
		Model::predict()

		Purpose: Predict temporal relations for a set of notes

		@param notes: A list of TimeNote objects
		@return classes: The temporal relations identified by the classifiers
		'''
		return


	def _trainTimex(self, tokenVectors, Y):
		'''
		Model::_trainTimex()

		Purpose: Train a classifer for Timex3 expressions

		@param tokenVectors: A list of tokenized sentences
		@param Y: A list of lists of Timex3 classifications for each token in each sentence
		'''

		clf, vec = train_classifier(tokenVectors, Y)
		self.timexClassifier = clf
		self.timexVectorizer = vec


	def _trainEvent(self, tokenVectors, Y):
		'''
		Model::_trainEvent()

		Purpose: Train a classifer for event identification

		@param tokenVectors: A list of tokenized sentences
		@param Y: A list of lists of event classifications for each token, with one list per sentence
		'''

		clf, vec = train_classifier(tokenVectors, Y)
		self.eventClassifier = clf
		self.eventVectorizer = vec


	def _trainRelation(self, tokenVectors, timexes, events, Y):
		'''
		Model::_trainRelation()

		Purpose: Train a classifer for temporal relations between events and timex3 labels

		@param tokenVectors: a list of tokenized sentences
		@param timexes: a list of Timex3 labels
		@param events: a list of event labels
		@param Y: A list of relation classifications for each pair of timexes and events.
		'''
		return


	

