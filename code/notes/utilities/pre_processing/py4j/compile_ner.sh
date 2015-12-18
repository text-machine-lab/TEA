
COMPILE_DEST=$TEA_PATH/code/notes/utilities/pre_processing/py4j/


NER_JAR_PATH=$TEA_PATH/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-nerc-1.5.2.jar

TOK_JAR_PATH=$TEA_PATH/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-tok-1.8.2.jar

POS_JAR_PATH=$TEA_PATH/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-pos-1.4.1.jar

javac -cp ".:$NER_JAR_PATH:$TOK_JAR_PATH:$POS_JAR_PATH" Ner.java -d $COMPILE_DEST


