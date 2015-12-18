
NER_JAR_PATH=$TEA_PATH/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-nerc-1.5.2.jar

COMPILE_DEST=$TEA_PATH/code/notes/utilities/pre_processing/py4j/

javac -cp ":$NER_JAR_PATH" Ner.java -d $COMPILE_DEST


