
import subprocess
import os

os.environ["TEA_PATH"] = os.getcwd()
os.environ["PUNKT_PATH"] = os.environ["TEA_PATH"] + "/data/nltk_data/tokenizers/punkt/english.pickle"

from code.notes.TimeNote import TimeNote

# instantiate note object
note = TimeNote("wsj_1025.tml")

print "text body of timeml document"
print note._process()

print "vectorized note"
print note.vectorize()

