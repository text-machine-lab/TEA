
if [ "$PY4J_DIR_PATH" == "" ]; then
    echo "environment variable PY4J_DIR_PATH not specified"
    echo "please put exact path example .../venv/share/py4j"
    exit
fi

PY4J_DEPENDENCIES=":$PY4J_DIR_PATH/*"

COMPILE_DEST=$TEA_PATH/code/notes/utilities/pre_processing/py4j/

javac -cp ":$TEA_PATH/code/notes/utilities/pre_processing/py4j/" EntryPoint.java -d $COMPILE_DEST

javac -cp "$PY4J_DEPENDENCIES" GateWay.java -d $COMPILE_DEST

