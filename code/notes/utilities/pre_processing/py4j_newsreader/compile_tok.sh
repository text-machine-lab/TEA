
TOK_JAR_PATH=$TEA_PATH/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-tok-1.8.2.jar

COMPILE_DEST=$TEA_PATH/code/notes/utilities/pre_processing/py4j_newsreader/

SRC_DIR=$TEA_PATH/code/notes/utilities/pre_processing/py4j_newsreader

javac -cp ":$TOK_JAR_PATH" $SRC_DIR/TokCLI.java -d $COMPILE_DEST
javac -cp "$COMPILE_DEST:$TOK_JAR_PATH" $SRC_DIR/Tok.java -d $COMPILE_DEST

