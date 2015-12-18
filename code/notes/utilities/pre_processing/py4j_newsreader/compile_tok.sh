
TOK_JAR_PATH=$TEA_PATH/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-tok-1.8.2.jar

COMPILE_DEST=$TEA_PATH/code/notes/utilities/pre_processing/py4j_newsreader/

javac -cp ":$TOK_JAR_PATH" TokCLI.java -d $COMPILE_DEST
javac -cp ":$TOK_JAR_PATH" Tok.java -d $COMPILE_DEST

