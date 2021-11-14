import os

if __name__ == "__main__":
    print("executing pipeline")
    os.system("preprocess.py -input genes/ -output preprocessed/")
    os.system("classify.py -input preprocessed/")