# Iannwtf Hw3 documentation

@authors: llemke, hrichtert, kbaublys

## Overall structure

This repository contains 2 versions of the homework. The first one can
be found in the notebook subfolder. But because we wanted a version which is 
able to run on linux and windows!, we made a second version which can be found in the windows
subfolder. Due to the tfds.load command not working on windows, the second version has a slightly 
different structure than the notebook version, which will be eloborated in the coming paragraphs

## Colab

The notebook version can be found in the colab subdirectory.
This version consists of a single .ipynb file.

## windows

The windows version can be found in the windows subdirectory.
In contrast to the notebook version, this one consists 
of different .py files. 

Because tfds.load doesn't work on windows, we had to choose a slightly different structure.
We implemented a class called DataLoader, which can be used to save and load tensorflow datasets locally (on the machine).
In order to access the datasets on windows, we did the following:
  - load the datasets in colab
  - save them using the dataloader
  - download the directory where the datasets are saved
  - load the datasets on windows again using the dataloader

The raw_data folder contains the datasets that where saved in colab and downloaded.
After loading and preprocessing the data, the preprocessed datasets will be saved
in the preprocessed_data folder. This gives everything a better structure. 

After that, the datasets will be loaded from the preprocessed_data folder again and
used for training and testing the model. 

The combined step of preprocessing and classification will be called the pipeline 
of the project. Here is how to run everything:

### Pipeline

The whole pipeline can be run using one of these two options:
- pipeline.py

When running the whole pipeline, the working directory is not important,
because it will be automatically recognized and adapted in the file. 

### Preprocessing

Preprocessing can be run using
- preprocess.py -input INPUT_PATH -output OUTPUT_PATH

Here, INPUT_PATH is the path to the datasets which should be preprocessed and
OUTPUT_PATH the path to where the preprocessed datasets should be saved.
Both have to be specified.

### Classification

Classification can be run using
- classify.py -input INPUT_PATH -num_epochs NUM_EPOCHS

Here, INPUT_PATH is the path to the datasets which should be used
for training and testing. NUM_EPOCHS is the number of training
epochs. INPUT_PATH has to be specified. NUM_EPOCHS doesn't have 
to be specified. The default value is 10.