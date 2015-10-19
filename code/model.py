from machine_learning.sci import train as train_classifier

class Model:

	def train(self, notes):
		'''
		Model::train()

		Purpose: Train a classification model for Timex3, events, and temporal event relations from annotated data

		@param notes. A list of TimeNote objects containing annotated data
		'''

		#TODO: add extraction methods that actually get features from annotated data
		
		#train classifiers
		self._trainTimex(tokenizedSents, timexLabels)
		self._trainEvent(tokenizedSents, eventLabels)
		self._trainRelation(tokenizedSents, timexLabels, eventLabels, relationLabels)

	def predict(self, notes):
		'''
		Model::predict()

		Purpose: Predict temporal relations for a set of notes

		@param notes: A list of TimeNote objects
		@return classes: The temporal relations identified by the classifiers
		'''

	def _trainTimex(self, tokenizedSents, Y):
		'''
		Model::_trainTimex()

		Purpose: Train a classifer for Timex3 expressions

		@param tokenizedSents: A list of tokenized sentences
		@param Y: A list of lists of Timex3 classifications for each token in each sentence
		'''

	def _trainEvent(self, tokenizedSents, Y):

	def _trainRelation(self, tokenizedSents, timexes, events, Y):
		'''
		Model::_trainRelation()

		Purpose: Train a classifer for temporal relations between events and timex3 labels

		@param tokenizedSents: a list of tokenized sentences
		@param timexes: a list of Timex3 labels
		@param events: a list of event labels
		'''

	

