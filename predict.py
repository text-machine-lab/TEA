import os
import cPickle
import argparse

os.environ["TEA_PATH"] = os.getcwd()
os.environ["PUNKT_PATH"] = os.environ["TEA_PATH"] + "/data/nltk_data/tokenizers/punkt/english.pickle"

from code.notes.TimeNote import TimeNote
from code.notes.utilities.timeml_utilities import annotate_text_element
from code.notes.utilities.timeml_utilities import set_text_element
from code.notes.utilities.xml_utilities import get_root
from code.notes.utilities.xml_utilities import write_root_to_file
from code import model

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("model", help="Model file to use in prediction")
	parser.add_argument("predictList", metavar='Predict List', nargs='+', 
		help="The list of files to annotate.")
	args = parser.parse_args()
	
	modfile = args.model
	files = args.predictList

	#load data from files
	notes = []

	#read in files as notes
	for tml in files:
		tmp_note = TimeNote(tml)
		notes.append(tmp_note)

	with open(modfile) as modelfile:
		model = cPickle.load(modelfile)

	for note in notes:
		timexLabels, eventLabels, offsets = model.predict(note)
		writeAnnotatedFile(note, timexLabels, eventLabels, offsets)



def writeAnnotatedFile(note, timexLabels, eventLabels, offsets):

	root = get_root(note.note_path)

	length = len(offsets)

	for i in range(1, length):
		if(timexLabels[length - i] != "O"):
			if(timexLabels[length - i][0] == "B"):
				start = offsets[length - i][0]
				end = offsets[length - i][1]

				#grab any IN tokens and add them to the tag text
				for j in range (1, i):
					if(timexLabels[length - i + j][0] == "I"):
						end = offsets[length - i + j][1]
					else:
						break

				annotated_text = annotate_text_element(root, "TIMEX3", start, end, {"tid":"t" + str(length - i), "type":timexLabels[length - i][2:]})
				set_text_element(root, annotated_text)

		elif(eventLabels[length - i] != "O"):
			if(eventLabels[length - i][0] == "B"):
				start = offsets[length - i][0]
				end = offsets[length - i][1]

				#grab any IN tokens and add them to the tag text
				for j in range (1, i):
					if(eventLabels[length - i + j][0] == "I"):
						end = offsets[length - i + j][1]
					else:
						break

				annotated_text = annotate_text_element(root, "EVENT", start, end, {"eid":"e" + str(length - i), "class":eventLabels[length - i][2:]})
				set_text_element(root, annotated_text)

	write_root_to_file(root)	

if __name__ == '__main__':
	main()