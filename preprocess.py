import argparse
from tensorflow.python.ops.gen_array_ops import reshape
from dataloader import DataLoader
from util import prepare_data

parser = argparse.ArgumentParser(description='Specify the path from where the datasets should be loaded and where the preprocessed datasets should be stored')
parser.add_argument('-input', type=str, help = "Path to dataset folders")
parser.add_argument('-output', type = str, help = "Path to where the preprocessed datasets should be stored")

args = parser.parse_args()

# check if input path is given by the user
if args.input is None:
    Exception("Please specify the loading directory for the datasets with -input when running preprocess.py")
    
# check if output path is given by the user
if args.output is None:
    Exception("Please specify the saving directory for the datasets with -output when running preprocess.py")

# load the datasets from the given path
dl = DataLoader()
dl.load_data(args.input)
data = dl.get_data()

train_ds = data["train"]
test_ds = data["test"]

# apply preprocessing to the datasets
train_ds = train_ds.apply(prepare_data)
test_ds = test_ds.apply(prepare_data)

# save the datasets in the given path
dl.set_data({"train":train_ds,"test":test_ds})
dl.save_data(args.output)