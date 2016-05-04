#!/bin/bash
: <<'END'
Compile all necessary java dependencies to use timenorm

Requirements:
    *maven3
    *maven3 is using java 1.8
END




if [ "$1" == 'clean' ]; then

    rm -rf timenorm *.class

elif [ "$1" == 'test' ]; then

    scala -cp ".:$(pwd)/timenorm/target/timenorm-0.9.1-SNAPSHOT.jar" TimeNorm "04/26/1993"

else

    tar -zxvf timenorm-timenorm-0.9.5.tar.gz

    cd timenorm-timenorm-0.9.5
    mvn install
    cd ../
    mv timenorm-timenorm-0.9.5/target/timenorm-0.9.5.jar ./

    # scalac -cp ".:$(pwd)/timenorm/target/timenorm-0.9.1-SNAPSHOT.jar" TimeNorm.scala

fi

