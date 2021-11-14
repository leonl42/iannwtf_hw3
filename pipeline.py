import os

if __name__ == "__main__":
    print("executing pipeline")
    os.system("preprocess.py -input raw_data/ -output preprocessed_data/")
    os.system("classify.py -input preprocessed_data/ -num_epochs 10")