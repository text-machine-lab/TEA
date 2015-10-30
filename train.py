import subprocess
import os
import cPickle

os.environ["TEA_PATH"] = os.getcwd()
os.environ["PUNKT_PATH"] = os.environ["TEA_PATH"] + "/data/nltk_data/tokenizers/punkt/english.pickle"

from code.notes.TimeNote import TimeNote

from code import model

def main():

	model = trainModel(["wsj_1025.tml"])

	with open("models/test.mod", "wb") as modFile:
		cPickle.dump(model, modFile)


def trainModel( training_list ):
	'''
	train::trainModel()

	Purpose: Train a model for classification of events, timexes, and temporal relations based 
			 on given training data

	@param training_list: List of strings containing file paths for .tml training documents
	'''

	print "Called train"

	# Read in notes
	notes = []

	for tml in training_list:
		tmp_note = TimeNote(tml)
		notes.append(tmp_note)

	mod = model.Model()
	mod.train(notes)

	print mod
	return mod

if __name__ == "__main__":
	main()