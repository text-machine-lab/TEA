QA TempEval Kit (TEST)
---------------------

With this kit you can test your system against data.


Requirements
------------
OS Linux or Mac (if you use windows read the notes for windows below)
Java 1.7

Folder structure
----------------

Input data given to participants to annotate:
data/
	input/
		file1
		...
		fileN
	
participants must annotate the input files using their systems and store the output annotation in:
systems/
	SystemName-RunName/
		file1
		..
		fileN

	SystemName-RunNameOther/
		file1
		..
		fileN

	...

Evaluation questions and the key event ids:
eval-questions/
	question-set.txt
	question-events/
		file1
		..
		fileN

A folder with the packaged version of the tools:
tools/
	timeml-normalizer/*
	timeml-qa/
* system annotations event ids should be normalized to match the event ids used in the questions.

	
When you run evaluate_systems.sh it will output results and logs:
results/
	... (system specific logs and results)
	log.txt (Check that there are no errors)
	results.txt (Same format/information as the official results)
	

Windows instructions:
--------------------

1. Manually run the normalization of your system against question events with the option "-respect"

java -jar tools/timeml-normalizer/timeml-normalizer-1.1.0.jar -a "PATH-TO-eval-questions/question-events/;PATH-TO-SYSTEM-ANNOTATIONS" -respect

2. Copy the question-set.txt in your system's annotation folder

3. Run timeml-qa to obtain the results
java -jar tools/timeml-qa/timeml-qa-1.0.0.jar -a TQA PATH-TO-SYSTEM-ANNOTATIONS-normalized/question-set.txt




