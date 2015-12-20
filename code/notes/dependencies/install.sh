
# IMPORTANT!!!
# make sure that mvn uses java 1.8, this can done by seting JAVA_HOME within .mavenrc
# make sure scala version matches pom.xml version.

if [ "$1" == 'clean' ]; then

    rm -rf timenorm *.class

elif [ "$1" == 'test' ]; then

    scala -cp ".:$(pwd)/timenorm/target/timenorm-0.9.1-SNAPSHOT.jar" TimeNorm "04/26/1993"

else

    tar -xvf timenorm.tar

    mvn clean install -f timenorm/pom.xml

    scalac -cp ".:$(pwd)/timenorm/target/timenorm-0.9.1-SNAPSHOT.jar" TimeNorm.scala


fi

