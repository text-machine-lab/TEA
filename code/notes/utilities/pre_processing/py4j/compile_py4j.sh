
if [ "$PY4J_DIR_PATH" == "" ]; then
    echo "environment variable PY4J_DIR_PATH not specified"
    echo "please put exact path example .../venv/share/py4j"
    exit
fi

PY4J_DEPENDENCIES=":$PY4J_DIR_PATH/*"

javac -cp "." EntryPoint.java -d .

javac -cp "$PY4J_DEPENDENCIES" GateWay.java -d .

