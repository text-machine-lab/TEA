#!/bin/bash

# input: path to data folder (with participants annotations one folder per run), if omitted takes script path as default
# note: the questionset and question-events annotation will be in the same path + "_eval-questions"

if [[ $# -le 0 ]];then
	echo "ERROR: One or more tml files need to be provided as arguments"
	exit 1
fi

for ifile in $@;do
	if [[ ! -f "$ifile" ]]; then
    		echo "ERROR: File not found ($ifile)!"
    		continue
	fi
	folder=$(dirname "${ifile}")
	name=$(basename "${ifile}")
	grep "<DCT>" "$ifile"
	dct=`grep "<DCT>" "$ifile" | sed "s/\//--slash--/g" | sed "s/\"/--quote--/g"`
	cat "$ifile" | sed "s/<\(TLINK\|MAKEINSTANCE\).*//g" | sed "s/<[\/]\?\(TIMEX\|SIGNAL\)[^>]*>//g" | sed "s/[[:blank:]]*\(class\|pos\|tense\|aspect\|polarity\|modality\)=\"[^\"]*\"[[:blank:]]*//g" | sed "s/^.*<DCT>.*\$/$dct/" | sed "s/--quote--/\"/g;s/--slash--/\//g"  > "$folder/readable.$name"
	sed -i "s/[[:blank:]]*\(temporalFunction\|functionInDocument\)=\"[^\"]*\"//g" "$folder/readable.$name"
	echo "created: $folder/readable.$name"
done



