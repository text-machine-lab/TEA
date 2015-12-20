#!/bin/bash

IFS="
"

SCRIPT_PATH=$(dirname "$(readlink "${BASH_SOURCE[0]}")") # -e option removed since it does not work in MAC
PARTICIPANTION_PATH=$SCRIPT_PATH/
if [[ $# -ne 1 || ! -d $SCRIPT_PATH/results/$1 ]];then
	echo "error: timestamp e.g., 2015-01-15T0050 required and must be a folder in results/"; exit 1
fi

curr_timestamp=$1
mkdir -p $SCRIPT_PATH/results/$curr_timestamp-analysis


echo "Using $PARTICIPANTION_PATH as path to find participations." 
echo -e "RESULTS\n-------\nsystem_name\t\tquests\tanswrd\tcorr\tinc\tcov\tprec\trec\tf1\tds\tnae\tunkent\tunkrel" > $SCRIPT_PATH/results/$curr_timestamp-analysis/results.txt

# >>>>> Normalize participant TimeML with question-events

	# For each folder
	for folder in `ls -d ${PARTICIPANTION_PATH}systems/*/`;do
		if [[ "$folder" != "${PARTICIPANTION_PATH}systems/input/" && $folder != *-normalized* ]];then
			echo -e "\n>>>> Processing $folder"
			system_run_name=`echo $folder | sed -e "s/.*\/\(.*\)\/\$/\1/"`;
			ds=0
			nae=0
			unkent=0
			unkrel=0
			for line in `cat $SCRIPT_PATH/results/$curr_timestamp/$system_run_name-results.txt`;do
				#echo "$line"
				if [ `echo $line | grep -v "^[0-9]\+|"` ]; then echo "nooo $line";continue; fi 
				qid=`echo $line | cut -f 1 -d '|'`
				doc=`echo $line | cut -f 2 -d '|'`
				question=`echo $line | cut -f 3 -d '|'`
				expected=`echo $line | cut -f 5 -d '|'`
				predicted=`echo $line | cut -f 6 -d '|' | sed "s/predicted=//"`
				#echo "$doc $question $expected $predicted"
				if [ "$expected" == "$predicted" ];then #correct
					sentsplit=`cat $SCRIPT_PATH/eval-questions/question-events/$doc | sed "s/[.!?:;]\+ /\n/g"`
					qent1=`echo $line | cut -d " " -f 2`
					qent2=`echo $line | cut -d " " -f 4 | cut -d "|" -f 1`
					qent1sent=`echo -e "$sentsplit" | grep -n "\"$qent1\"" | cut -d ":" -f 1`
					qent2sent=`echo -e "$sentsplit" | grep -n "\"$qent2\"" | cut -d ":" -f 1`
					qdist=$((qent2sent - qent1sent))
					if [ $qdist -le -1 ];then #qdist=${qdist%-}; 
						qdist=$((0 - qdist))
					fi			
					if [ "$qdist" == "1" -o "$qdist" == "0" ];then
						echo "close"					
					else
						ds=$((ds + 1))					
					fi
					#others?
					notfound=0
					for folder2 in `ls -d ${PARTICIPANTION_PATH}systems/*/`;do
						system_run_name2=`echo $folder2 | sed -e "s/.*\/\(.*\)\/\$/\1/"`;
						notrefl=`echo $system_run_name | sed "s/-timex-ref-links//"`;
						if [[ "$system_run_name2" != "${PARTICIPANTION_PATH}systems/input/" && $system_run_name2 != *-normalized* && $system_run_name2 != *$notrefl* ]];then
							echo "comparing with $system_run_name2 $notrefl"
							line2=`grep "^${qid}|" $SCRIPT_PATH/results/$curr_timestamp/$system_run_name2-results.txt`
							predicted2=`echo $line2 | cut -f 6 -d '|' | sed "s/predicted=//"`
							if [ "$expected" == "$predicted2" ];then
								echo "question $qid also answered by $system_run_name2"
								notfound=1;break
							fi
						fi
					done
					if [ $notfound -eq 0 ];then
						nae=$((nae + 1))
						echo "question $qid only answered by $system_run_name"
					fi
					echo "$question -> correct qdist=$qdist ds=$ds nae=$nae"
				else #incorrect/unknown
					if [[ $predicted == *"unknown"* ]]; then
						if [[ $predicted == *"entity"* ]]; then
							unkent=$((unkent + 1))		
						else
							unkrel=$((unkrel + 1))					
						fi					
					fi
				fi
			done
			results=`cat $SCRIPT_PATH/results/$curr_timestamp/$system_run_name-results.txt | grep "^questions" | sed -e "s/[^=]*=\([^[:blank:]]*\)/	\1/g"` # Mac sed does not support \t so we use a normal tab
			echo -e "${system_run_name}\t$results\t$ds\t$nae\t$unkent\t$unkrel" >> $SCRIPT_PATH/results/$curr_timestamp-analysis/results.txt
		fi
	done
	echo -e "\n\n--------------------------------------------"
	cat $SCRIPT_PATH/results/$curr_timestamp-analysis/results.txt
	echo "--------------------------------------------"
	echo -e "\n\nRESULTS: $SCRIPT_PATH/results/$curr_timestamp-analysis/results.txt\n"
	



