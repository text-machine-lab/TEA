#!/bin/bash
: <<'END'
Compile all necessary java dependencies for usage by python system component.
END

# gets current directory where script is located.
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# base directory of project.
TEA_HOME=$CUR_DIR
TEA_HOME+="/../../../../../"

export TEA_PATH=$TEA_HOME

# process config.txt to get PY4J_DIR_PATH
config=$(cat "$TEA_HOME/config.txt")
readarray -t config<<<"$config"

for l in "${config[@]}";
do
    l=($l)

    option=${l[0]}
    setting=${l[1]}

    if [ "$option" == "PY4J_DIR_PATH" ]; then
        PY4J_DIR_PATH=$setting
    fi

done

if [ "$PY4J_DIR_PATH" == "" ]; then
    echo "environment variable PY4J_DIR_PATH not specified in config.txt"
    echo "please put exact path example .../venv/share/py4j"
    exit
fi

export PY4J_DIR_PATH=$PY4J_DIR_PATH

bash $TEA_HOME/code/notes/utilities/pre_processing/py4j_newsreader/compile_tok.sh
bash $TEA_HOME/code/notes/utilities/pre_processing/py4j_newsreader/compile_pos.sh
bash $TEA_HOME/code/notes/utilities/pre_processing/py4j_newsreader/compile_ner.sh
bash $TEA_HOME/code/notes/utilities/pre_processing/py4j_newsreader/compile_parse.sh

# TODO: move this into a seperate directory since now there is stuff outside of this that depends on it.
bash $TEA_HOME/code/notes/utilities/pre_processing/py4j_newsreader/compile_py4j.sh

unset TEAH_PATH
unset PY4J_DIR_PATH

