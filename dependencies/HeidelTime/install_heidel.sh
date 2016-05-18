#!/bin/bash
: <<'END'
Compile all necessary java dependencies for usage by python system component.
END

# gets current directory where script is located.
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# base directory of project.
TEA_HOME=$CUR_DIR
TEA_HOME+="/../../"

HEIDELTIME_PATH=$TEA_HOME"/dependencies/HeidelTime/"
TREETAGGER_PATH=$TEA_HOME"dependencies/HeidelTime/treetagger"

# install heideltime
tar -xvf heideltime-kit-2.1.tar.gz -C $HEIDELTIME_PATH
cd $HEIDELTIME_PATH"/heideltime-kit"
mvn clean install
cd ..

# install treetagger used by heideltime
mkdir $TREETAGGER_PATH
cd $TREETAGGER_PATH

wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-linux-3.2.tar.gz
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tagger-scripts.tar.gz
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/install-tagger.sh
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/english-par-linux-3.2-utf8.bin.gz
bash install-tagger.sh
cd ..

# set treetagger path for heideltime
sed -i 's|treeTaggerHome.*|treeTaggerHome = '$TREETAGGER_PATH'|g' $HEIDELTIME_PATH"/heideltime-kit/conf/config.props"

