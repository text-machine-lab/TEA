
PARSE_JAR_PATH=$TEA_PATH/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-parse-1.1.0.jar

COMPILE_DEST=$TEA_PATH/code/notes/utilities/pre_processing/py4j/

javac -cp ":$PARSE_JAR_PATH" Parse.java -d $COMPILE_DEST

