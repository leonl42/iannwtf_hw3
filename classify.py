import tensorflow as tf
from model import MyModel
from dataloader import DataLoader
from util import train_step, test, visualize
import argparse

parser = argparse.ArgumentParser(description='Paths for files')
parser.add_argument('-input', type=str, help = "Path to input files")
parser.add_argument('-num_epochs', type=int, help = "number of epochs", default=100)

args = parser.parse_args()

dl = DataLoader()
dl.load_data("preprocessed/")
data = dl.get_data()
train_ds = data["train"]
test_ds = data["test"]

tf.keras.backend.clear_session()

# Hyperparameters
num_epochs = args.num_epochs
learning_rate = 0.1

# Initialize the model.
model = MyModel()
# Initialize the loss: categorical cross entropy
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
# Initialize the optimizer: SGD with default parameters
optimizer = tf.keras.optimizers.SGD(learning_rate)

# Initialize lists for later visualization.
train_losses = []
test_losses = []
test_accuracies = []

# Testing on our test_ds once before we begin
test_loss, test_accuracy = test(model, test_ds, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

# Testing on our train_ds once before we begin
train_loss, _ = test(model, train_ds, cross_entropy_loss)
train_losses.append(train_loss)

# Training our modelfor num_epochs epochs.
for epoch in range(num_epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

    # training (and calculating loss while training)
    epoch_loss_agg = []
    for input,target in train_ds:
        train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)
    
    # track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    # testing our model in each epoch to track accuracy and test loss
    test_loss, test_accuracy = test(model, test_ds, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

print("visualizing...")
visualize(train_losses,test_losses,test_accuracies)