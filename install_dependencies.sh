
TEA_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ "$PY4J_DIR_PATH" == "" ]; then
    echo "environment variable PY4J_DIR_PATH not specified"
    echo "please put exact path example .../venv/share/py4j"
    exit
fi

# unpack dependencies
tar -xvf NewsReader.tar

# installing srl dependencies
wget https://storage.googleapis.com/google-code-archive-downloads/v2/src.google.com/mate-tools/srl-4.3.tgz

tar -zxvf srl-4.3.tgz srl-20130917/lib/liblinear-1.51-with-deps.jar
tar -zxvf srl-4.3.tgz srl-20130917/lib/seg.jar
tar -zxvf srl-4.3.tgz srl-20130917/srl.jar

mvn install:install-file -Dfile=srl-20130917/lib/liblinear-1.51-with-deps.jar -DgroupId=local -DartifactId=liblinear -Dversion=1.51 -Dpackaging=jar
mvn install:install-file -Dfile=srl-20130917/srl.jar -DgroupId=local -DartifactId=srl -Dversion=1.0 -Dpackaging=jar
mvn install:install-file -Dfile=srl-20130917/lib/seg.jar -DgroupId=local -DartifactId=seg -Dversion=1.0 -Dpackaging=jar

# extract srl tar ball
tar -xvf NewsReader/ixa-pipes-1.1.0/ixa-pipe-srl.tar -C NewsReader/ixa-pipes-1.1.0/

# JAVA 1.7 or higher worked fine. ensure maven is using right version.
mvn clean install -f NewsReader/ixa-pipes-1.1.0/ixa-pipe-srl/IXA-EHU-srl/pom.xml

wget https://storage.googleapis.com/google-code-archive-downloads/v2/src.google.com/mate-tools/CoNLL2009-ST-English-ALL.anna-3.3.parser.model --directory-prefix=NewsReader/ixa-pipes-1.1.0/ixa-pipe-srl/IXA-EHU-srl/target/models/eng/
wget http://fileadmin.cs.lth.se/nlp/models/srl/en/srl-20100906/srl-eng.model --directory-prefix=NewsReader/ixa-pipes-1.1.0/ixa-pipe-srl/IXA-EHU-srl/target/models/eng/

mkdir NewsReader/ixa-pipes-1.1.0/ixa-pipe-srl/IXA-EHU-srl/target/PredicateMatrix

wget http://adimen.si.ehu.es/web/files/PredicateMatrix/PredicateMatrix.srl-module.tar.gz
tar -zxf PredicateMatrix.srl-module.tar.gz -C NewsReader/ixa-pipes-1.1.0/ixa-pipe-srl/IXA-EHU-srl/target/PredicateMatrix/
rm -f PredicateMatrix.srl-module.tar.gz

rm -rf dependencies/NewsReader
mv NewsReader dependencies/

. $TEA_HOME/code/notes/utilities/pre_processing/py4j_newsreader/compile.sh

echo "finished installation"

