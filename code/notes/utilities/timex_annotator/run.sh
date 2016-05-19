#!/bin/bash
: <<'END'
Compile all necessary java dependencies for usage by python system component.
END

# gets current directory where script is located.
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# base directory of project.
TEA_HOME=$CUR_DIR
TEA_HOME+="/../../../../"

HEIDELTIME_PATH=$TEA_HOME"/dependencies/HeidelTime/"

DEPENDENCIES=":$HEIDELTIME_PATH/heideltime-kit/lib/*:$HEIDELTIME_PATH/heideltime-kit/target/*"

JAVA_SRC=$TEA_HOME"/code/notes/utilities/timex_annotator/HeidelTime.java"

java -cp $DEPENDENCIES heidel.HeidelTime

