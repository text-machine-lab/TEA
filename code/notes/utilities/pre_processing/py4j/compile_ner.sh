
NER_JAR_PATH=$TEA_PATH/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-nerc-1.5.2.jar

javac -cp ".:$NER_JAR_PATH" Ner.java -d .

#java -cp ".:ixa-pipe-tok-1.8.2.jar" eus.ixa.ixa.pipe.tok.Tok


