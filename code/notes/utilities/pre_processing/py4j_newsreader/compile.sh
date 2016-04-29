#!/bin/bash
: <<'END'
Compile all necessary java dependencies for usage by python system component.
END




if [ "$PY4J_DIR_PATH" == "" ]; then
    echo "environment variable PY4J_DIR_PATH not specified"
    echo "please put exact path example .../venv/share/py4j"
    exit
fi

if [ "$TEA_PATH" == "" ]; then
    echo "environment variable TEA_PATH not specified"
    exit
fi

bash $TEA_PATH/code/notes/utilities/pre_processing/py4j_newsreader/compile_tok.sh
bash $TEA_PATH/code/notes/utilities/pre_processing/py4j_newsreader/compile_pos.sh
bash $TEA_PATH/code/notes/utilities/pre_processing/py4j_newsreader/compile_ner.sh
bash $TEA_PATH/code/notes/utilities/pre_processing/py4j_newsreader/compile_parse.sh
bash $TEA_PATH/code/notes/utilities/pre_processing/py4j_newsreader/compile_py4j.sh

