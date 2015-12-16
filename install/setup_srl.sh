
tar -xvf ixa-pipe-srl.tar

mvn clean install -f ixa-pipe-srl/IXA-EHU-srl/pom.xml

wget http://mate-tools.googlecode.com/files/CoNLL2009-ST-English-ALL.anna-3.3.parser.model --directory-prefix=ixa-pipe-srl/IXA-EHU-srl/target/models/eng/
wget http://fileadmin.cs.lth.se/nlp/models/srl/en/srl-20100906/srl-eng.model --directory-prefix=ixa-pipe-srl/IXA-EHU-srl/target/models/eng/

mkdir ixa-pipe-srl/IXA-EHU-srl/target/PredicateMatrix

wget http://adimen.si.ehu.es/web/files/PredicateMatrix/PredicateMatrix.srl-module.tar.gz
tar -zxf PredicateMatrix.srl-module.tar.gz -C ixa-pipe-srl/IXA-EHU-srl/target/PredicateMatrix/
rm -f PredicateMatrix.srl-module.tar.gz


