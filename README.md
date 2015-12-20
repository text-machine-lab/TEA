# nlproject
NLP Project

How to install:

    1. Download the latest release from:

        https://github.com/ktwaco/Temporal-Entity-Annotator-TEA-/releases

    2. Also download the NewsReader.tar file from the release.

    3. Uncompress NewsReader.tar

        A folder called Temporal-Entity-Annotator-TEA-VERSION should be created

    4. Place NewsReader.tar within the extracted folder.

    5. Change current working directory to be the extracted folder.

    6. Execute the bash script install_dependencies.sh

External Dependencies:

    - maven3
    - java1.7 or higher
    - python2.7
    - py4j, python module
    - scikitlearn

Environment Variables:

    1. There are two that need to be defined for the system to work:
        - TEA_PATH, should be set to the the path .../Temporal-Entity-Annotator-TEA-VERSION

        - PY4J_DIR_PATH, should be set to the folder /share/py4j, created when installing py4j.
            - It should contain the contain the file py4j0.8.2.1.jar


