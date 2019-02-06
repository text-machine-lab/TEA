# Temporal Entity Annotator on Timebank-Dense data


The code in this repository was used in the following publication:  Y. Meng and A. Rumshisky,  Context-Aware Neural Model for Temporal Information Extraction http://aclweb.org/anthology/P18-1049 

You may contact ylmeng for any issues.

Copyright (c) 2018 UMass Lowell - All Rights Reserved

You may use, distribute and modify this code under the terms of the Apache 2.0 license. See https://www.apache.org/licenses/LICENSE-2.0 for details

## How to install:

    1. checkout the "py36" branch (note: not the "master" branch):
       
    2. Uncompress "newsreader_notes.tar.gz". Make sure you have all the pickle files.
    
    3. Uncompress "training-dev.tar.gz". It will create a folder containing training files and validation files. You may want to make separate folders for training and validation. Just create new folders and copy the files.
    
    4. Uncompress "test.tar.gz". It will create a folder containing test files.

Note: The training and test files do NOT contain the correct T-Links for Timebank-Dense. They only serve to provide text and event/timex tags. The correct tinlks are found in "dense-labels.pkl" after you extract the "newsreader_notes.tar.gz" file.

Environment Variables:

    There are two environment variables that need to be defined for the system to work:
       - TEA_PATH, should be set to where you install this package.

       - PY4J_DIR_PATH, not really needed here. Could be set to current directory.
          create config.txt in your TEA path. Add this line to the file (change the path to yours):
           PY4J_DIR_PATH ./

Tensorflow and Keras:
    
    We tested our model on Tensorflow 1.7.0 and Keras 2.1.5, with python 3.6. Training and testing are performed on GPU-enabled computers.

## Data Sets:

If you uncompressed "newsreader_notes.tar.gz" successfully, you should see all the pickle files in folder newsreader_annotations/dense-7-6-18/. They are preprocessed data files, and have tokens, pairs and all the features we need.

In addition to the preprocessed data files, "dense-labels-single.pkl" collects all the labels for legitimate entity pairs, and "dense-labels.pkl" collects labels for the flipped pairs too. We will use "dense-labels.pkl".

## How to use:

### To replicate our results

Run the following command in TEA direcotry:

    $python train_gcl.py training-dev/ model_destination/ newsreader_annotations/dense-7-6-18/ --val_dir test/

training-dev/ -- folder containing all training and development files. You can exclude dev files for validation purpose; but for the final test, it is better to use them for training, because the training set is small.

model_destination/ -- where you save the models.

newsreader_annotations/dense-7-6-18/ --  folder containing preprocessed files. You will have it after installation.

test/ --  folder of test files.

After the system finishes training, evaluation will be performed on test set automatically.
    
### To run preprocessing
If you want to run the preprocessing procedure by yourself, instead of using the pickled files, you need to install other packages. Please refer to the README.m file on the master branch for details. After everything is installed, you can just point "newsreader_annotations" to an empty folder. Then the pkl files will be generated.


