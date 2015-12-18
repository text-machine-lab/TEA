
if [ "$PY4J_DIR_PATH" == "" ]; then
    echo "environment variable PY4J_DIR_PATH not specified"
    echo "please put exact path example .../venv/share/py4j"
    exit
fi

if [ "$TEA_PATH" == "" ]; then
    echo "environment variable TEA_PATH not specified"
    exit
fi


bash compile_tok.sh
bash compile_pos.sh
bash compile_ner.sh
bash compile_parse.sh
bash compile_py4j.sh


