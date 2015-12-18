
POS_JAR_PATH=$TEA_PATH/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-pos-1.4.1.jar

COMPILE_DEST=$TEA_PATH/code/notes/utilities/pre_processing/py4j_newsreader/

javac -cp ":$POS_JAR_PATH" PosCLI.java -d $COMPILE_DEST

javac -cp ":$POS_JAR_PATH" Pos.java -d $COMPILE_DEST

