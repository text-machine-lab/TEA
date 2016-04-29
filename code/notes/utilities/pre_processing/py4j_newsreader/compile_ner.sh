#!/bin/bash
: <<'END'
Compile newsreader NER pipeline component.
END




COMPILE_DEST=$TEA_PATH/code/notes/utilities/pre_processing/py4j_newsreader/
NER_JAR_PATH=$TEA_PATH/code/notes/dependencies/NewsReader/ixa-pipes-1.1.0/ixa-pipe-nerc-1.5.2.jar
SRC_DIR=$TEA_PATH/code/notes/utilities/pre_processing/py4j_newsreader

javac -cp ":$NER_JAR_PATH" $SRC_DIR/NerCLI.java -d $COMPILE_DEST
javac -cp ":$SRC_DIR:$NER_JAR_PATH" $SRC_DIR/Ner.java -d $COMPILE_DEST

