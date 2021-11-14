import os

if __name__ == "__main__":

    # set the working directory to the file directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    os.system("preprocess.py -input raw_data/ -output preprocessed_data/")
    os.system("classify.py -input preprocessed_data/ -num_epochs 10")