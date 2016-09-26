# create a virtualenv to run TEA in
cd ../
virtualenv venv_TEA -p python3
source venv_TEA/bin/activate

# install required python packages
pip install numpy
pip install scipy
pip install sklearn
pip install nltk
pip install py4j
pip install keras
pip install gensim
pip install h5py

# set required environment variables
export PY4J_DIR_PATH=$PWD'/venv_TEA/share/py4j'
