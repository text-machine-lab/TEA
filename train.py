import os
import cPickle
import argparse

os.environ["TEA_PATH"] = os.getcwd()
os.environ["PUNKT_PATH"] = os.environ["TEA_PATH"] + "/data/nltk_data/tokenizers/punkt/english.pickle"

from code.notes.TimeNote import TimeNote

from code import model

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("trainList", metavar='Training List', nargs='+', 
		help="The list of .tml files to train data on. It is assumed that every .tml file has a coresponding .tml.TE3input file clean of annotations.")
	args = parser.parse_args()
	
	files = args.trainList

	model = trainModel(files)

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
		tmp_note = TimeNote(tml + '.TE3input', tml)
		notes.append(tmp_note)

	mod = model.Model()
	mod.train(notes)

	return mod

if __name__ == "__main__":
	main()