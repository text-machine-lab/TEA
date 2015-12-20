#!/bin/bash

# Requeriments: Linux/Mac, Java 1.7

# input: path to data folder (with participants annotations one folder per run), if omitted takes script path as default
# note: the questionset and question-events annotation will be in the same path + "_eval-questions"

SCRIPT_PATH=$(dirname "$(readlink "${BASH_SOURCE[0]}")") # -e option removed since it does not work in MAC
PARTICIPANTION_PATH=$SCRIPT_PATH/
if [[ $# -gt 0 && -d $1 ]];then
	PARTICIPANTION_PATH=$1
fi

mkdir -p $SCRIPT_PATH/results/
curr_timestamp=`date +%Y-%m-%dT%H%M`
mkdir -p $SCRIPT_PATH/results/$curr_timestamp


echo "Using $PARTICIPANTION_PATH as path to find participations to evaluate." | tee $SCRIPT_PATH/results/$curr_timestamp/log.txt
echo -e "RESULTS\n-------\nsystem_name\t\tquests\tanswrd\tcorr\tinc\tacc\tprec\trec\tf1" > $SCRIPT_PATH/results/$curr_timestamp/results.txt

# >>>>> Normalize participant TimeML with question-events

	# For each folder
	for folder in `ls -d ${PARTICIPANTION_PATH}systems/*/`;do
		if [[ "$folder" != "${PARTICIPANTION_PATH}systems/input/" && $folder != *-normalized* ]];then
			echo -e "\n>>>> Processing $folder" | tee -a $SCRIPT_PATH/results/$curr_timestamp/log.txt
			system_run_name=`echo $folder | sed -e "s/.*\/\(.*\)\/\$/\1/"`;
			#rm -rf $folder-normalized; #not needed it overwrites
			# For each file, normalize with the analogue in _eval-questionset,
			#store the files in an analogue folder prefixed/sufixed with normalized_ ...
			echo -e "\tNormalizing event ids with eval questions (question events)"
			java -jar tools/timeml-normalizer/timeml-normalizer-1.1.0.jar -a "${PARTICIPANTION_PATH}eval-questions/question-events/;$folder" -respect >> $SCRIPT_PATH/results/$curr_timestamp/log.txt 2>&1
			#rm -rf ${PARTICIPANTION_PATH}eval-questions/question-events-normalized/ # this file should not be created with -respect option
			# >>>>> Evaluate participants normalized annotations against the question set and produce a readable output
				# For each folder with the "normalized" prefix/suffix
					# copy the questionset inside the folder
					# run Java TMQA over that question set in that path
					# print the results in a readable/parseable way
					# remove temp files
			cp ${PARTICIPANTION_PATH}eval-questions/question-set.txt ${folder%?}-normalized/question-set.txt
			echo -e "\tEvaluating the system against the question-set"
			java -jar tools/timeml-qa/timeml-qa-1.0.2.jar  -a TQA ${folder%?}-normalized/question-set.txt 2>> $SCRIPT_PATH/results/$curr_timestamp/log.txt > $SCRIPT_PATH/results/$curr_timestamp/$system_run_name-results.txt # -d
			echo -e "\tOutput: $SCRIPT_PATH/results/$curr_timestamp/$system_run_name-results.txt"
			#rm -rf ${folder%?}-normalized
			results=`cat $SCRIPT_PATH/results/$curr_timestamp/$system_run_name-results.txt | grep "^questions" | sed -e "s/[^=]*=\([^[:blank:]]*\)/	\1/g"` # Mac sed does not support \t so we use a normal tab
			echo -e "${system_run_name}\t$results" >> $SCRIPT_PATH/results/$curr_timestamp/results.txt
			rm -rf ${folder%?}-normalized/question-set.txt
		fi
	done
	echo -e "\n\n--------------------------------------------"
	cat $SCRIPT_PATH/results/$curr_timestamp/results.txt
	echo "--------------------------------------------"
	echo -e "\n\nLOG:$SCRIPT_PATH/results/$curr_timestamp/log.txt\nRESULTS: $SCRIPT_PATH/results/$curr_timestamp/results.txt\n"




