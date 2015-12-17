
# unpack dependencies
tar -xvf NewsReader.tar

# extract srl tar ball
tar -xvf NewsReader/ixa-pipes-1.1.0/ixa-pipe-srl.tar -C NewsReader/ixa-pipes-1.1.0/

mvn clean install -f NewsReader/ixa-pipes-1.1.0/ixa-pipe-srl/IXA-EHU-srl/pom.xml

wget http://mate-tools.googlecode.com/files/CoNLL2009-ST-English-ALL.anna-3.3.parser.model --directory-prefix=NewsReader/ixa-pipes-1.1.0/ixa-pipe-srl/IXA-EHU-srl/target/models/eng/
wget http://fileadmin.cs.lth.se/nlp/models/srl/en/srl-20100906/srl-eng.model --directory-prefix=NewsReader/ixa-pipes-1.1.0/ixa-pipe-srl/IXA-EHU-srl/target/models/eng/

mkdir NewsReader/ixa-pipes-1.1.0/ixa-pipe-srl/IXA-EHU-srl/target/PredicateMatrix

wget http://adimen.si.ehu.es/web/files/PredicateMatrix/PredicateMatrix.srl-module.tar.gz
tar -zxf PredicateMatrix.srl-module.tar.gz -C NewsReader/ixa-pipes-1.1.0/ixa-pipe-srl/IXA-EHU-srl/target/PredicateMatrix/
rm -f PredicateMatrix.srl-module.tar.gz

mv NewsReader code/notes/

