
POS_JAR_PATH=$TEA_PATH/code/notes/NewsReader/ixa-pipes-1.1.0/ixa-pipe-pos-1.4.1.jar

javac -cp ".:$POS_JAR_PATH" Pos.java -d .

#java -cp ".:ixa-pipe-tok-1.8.2.jar" eus.ixa.ixa.pipe.tok.Tok


