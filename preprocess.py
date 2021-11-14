import argparse
import numpy as np
from tensorflow.python.ops.gen_array_ops import reshape
from dataloader import DataLoader
import tensorflow as tf
from util import prepare_data

parser = argparse.ArgumentParser(description='Paths for files')
parser.add_argument('-input', type=str, help = "Path to input files")
parser.add_argument('-output', type = str, help = "Path to where the output files should be stored")

args = parser.parse_args()

dl = DataLoader()
dl.load_data(args.input)
data = dl.get_data()

train_ds = data["train"]
test_ds = data["test"]

train_ds = train_ds.apply(prepare_data)
test_ds = test_ds.apply(prepare_data)

dl.set_data({"train":train_ds,"test":test_ds})
dl.save_data(args.output)