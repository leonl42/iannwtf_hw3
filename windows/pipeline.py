import os

if __name__ == "__main__":

    # get directory of this file
    file_dir = os.path.dirname(__file__)

    os.system(f"{file_dir}/preprocess.py -input {file_dir}/raw_data/ -output {file_dir}/preprocessed_data/")
    os.system(f"{file_dir}/classify.py -input {file_dir}/preprocessed_data/ -num_epochs 10")