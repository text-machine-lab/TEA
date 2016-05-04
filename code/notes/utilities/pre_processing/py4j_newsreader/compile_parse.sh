#!/bin/bash
: <<'END'
Compile newsreader PARSE pipeline component.
END




PARSE_JAR_PATH=$TEA_PATH/dependencies/NewsReader/ixa-pipes-1.1.0/ixa-pipe-parse-1.1.0.jar
COMPILE_DEST=$TEA_PATH/code/notes/utilities/pre_processing/py4j_newsreader/
SRC_DIR=$TEA_PATH/code/notes/utilities/pre_processing/py4j_newsreader

javac -cp ":$PARSE_JAR_PATH" $SRC_DIR/ParseCLI.java -d $COMPILE_DEST
javac -cp ":$SRC_DIR:$PARSE_JAR_PATH" $SRC_DIR/Parse.java -d $COMPILE_DEST

